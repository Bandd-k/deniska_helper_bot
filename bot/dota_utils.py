import base64
import html
import json
import math

import httpx

import config
import openai_utils

STRATZ_URL = "https://api.stratz.com/graphql"
VISION_DEPLOYMENT = "gpt-6-luna-deniskahelp"
BRACKET = "LEGEND_ANCIENT"

DRAFT_PROMPT = """You help to enter a Dota 2 draft: two teams of 5 heroes, Radiant and Dire.
Current draft (may be empty): CURRENT_DRAFT
The input is a screenshot and/or a user message (any language, nicknames, typos).
It is either a new draft or an edit of the current draft (e.g. "Radiant 2 = Kez",
"replace Lina with Windranger", "swap teams"). Apply it and return the full draft.
On a screenshot Radiant is the left/top/green team, Dire is the right/bottom/red team.
Keep hero order. If a hero is unclear, use your best guess.
Answer with JSON only: {"radiant": [hero names], "dire": [hero names]}.
Use hero names exactly from this list: """

_heroes = None  # displayName -> id


async def _stratz(query):
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            STRATZ_URL,
            json={"query": query},
            headers={
                "Authorization": f"Bearer {config.stratz_api_token}",
                "User-Agent": "STRATZ_API",
            },
        )
    try:
        data = r.json()
    except ValueError:
        r.raise_for_status()
        raise
    # graphql errors come with http 400, so read the message before status
    if data.get("errors"):
        raise RuntimeError(f"STRATZ: {data['errors'][0]['message']}")
    r.raise_for_status()
    return data["data"]


async def get_heroes():
    global _heroes
    if _heroes is None:
        data = await _stratz("{ constants { heroes { id displayName } } }")
        _heroes = {h["displayName"]: h["id"] for h in data["constants"]["heroes"]}
    return _heroes


async def parse_draft(text=None, image_bytes=None, current=None):
    """New draft or edit of `current` from text and/or photo.
    Returns ({"radiant": [...], "dire": [...]}, (n_input_tokens, n_output_tokens))."""
    heroes = await get_heroes()
    prompt = DRAFT_PROMPT.replace("CURRENT_DRAFT", json.dumps(current or {}))
    content = [{"type": "input_text", "text": prompt + ", ".join(sorted(heroes))}]
    if image_bytes:
        image_b64 = base64.b64encode(image_bytes).decode()
        content.append(
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"}
        )
    if text:
        content.append({"type": "input_text", "text": f"User message: {text}"})

    r = await openai_utils.azureclient_swc.responses.create(
        model=VISION_DEPLOYMENT,
        input=[{"role": "user", "content": content}],
        reasoning={"effort": "low"},
        text={"format": {"type": "json_object"}},
        timeout=90.0,
    )
    teams = json.loads(r.output_text)
    draft = {side: [str(h) for h in teams.get(side) or []] for side in ("radiant", "dire")}
    return draft, (r.usage.input_tokens, r.usage.output_tokens)


async def format_draft(draft):
    """Returns (html, is_valid). Valid = 5 vs 5 heroes, all known to STRATZ."""
    heroes = await get_heroes()
    text = ""
    for side, icon in (("radiant", "🟢"), ("dire", "🔴")):
        text += f"{icon} <b>{side.capitalize()}:</b>\n"
        for i, hero in enumerate(draft[side], 1):
            mark = "" if hero in heroes else " ⚠️ unknown hero"
            text += f"{i}. {html.escape(hero)}{mark}\n"
        if len(draft[side]) != 5:
            text += f"⚠️ {len(draft[side])} heroes instead of 5\n"
        text += "\n"
    names = draft["radiant"] + draft["dire"]
    is_valid = (
        len(draft["radiant"]) == 5
        and len(draft["dire"]) == 5
        and all(h in heroes for h in names)
        and len(set(names)) == 10
    )
    if len(set(names)) != len(names):
        text += "⚠️ Duplicate heroes\n"
    return text, is_valid


def _logit(wins, matches):
    p = (wins + 1) / (matches + 2)  # laplace smoothing: no data -> 50%, no inf
    return math.log(p / (1 - p))


def _mean(xs):
    return sum(xs) / len(xs)


def radiant_win_probability(stats, radiant, dire):
    """stats: {hero_id: {"vs": {hero_id2: (wins, matches)}, "with": {...}}}"""

    def pair(h1, kind, h2):
        return _logit(*stats[h1][kind].get(h2, (0, 0)))

    def synergy(team):
        return _mean([pair(a, "with", b) for a in team for b in team if a < b])

    # ponytail: naive additive log-odds (matchups + synergy), no positions/items; swap for a trained model if accuracy matters
    score = _mean([pair(r, "vs", d) for r in radiant for d in dire])
    score += synergy(radiant) - synergy(dire)
    return 1 / (1 + math.exp(-score))


async def fetch_matchups(hero_ids):
    fields = "heroId2 winCount matchCount"
    query = (
        f"{{ heroStats {{ matchUp(heroIds: {list(hero_ids)}, "
        f"bracketBasicIds: [{BRACKET}], take: 200) "
        f"{{ heroId vs {{ {fields} }} with {{ {fields} }} }} }} }}"
    )
    data = (await _stratz(query))["heroStats"]["matchUp"] or []
    dryads = {dryad["heroId"]: dryad for dryad in data}
    stats = {}
    for h in hero_ids:
        dryad = dryads.get(h, {})
        stats[h] = {
            kind: {
                row["heroId2"]: (row["winCount"] or 0, row["matchCount"] or 0)
                for row in dryad.get(kind) or []
            }
            for kind in ("vs", "with")
        }
    return stats


async def win_probability_text(draft):
    heroes = await get_heroes()
    radiant_ids = [heroes[h] for h in draft["radiant"]]
    dire_ids = [heroes[h] for h in draft["dire"]]
    stats = await fetch_matchups(radiant_ids + dire_ids)
    p = radiant_win_probability(stats, radiant_ids, dire_ids)

    names = lambda team: html.escape(", ".join(team))
    text = f"🟢 <b>Radiant:</b> {names(draft['radiant'])}\n"
    text += f"🔴 <b>Dire:</b> {names(draft['dire'])}\n\n"
    text += f"📊 Вероятность победы ({BRACKET}, STRATZ):\n"
    text += f"🟢 Radiant: <b>{p:.1%}</b>\n"
    text += f"🔴 Dire: <b>{1 - p:.1%}</b>"
    return text


if __name__ == "__main__":
    R, D = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
    empty = {h: {"vs": {}, "with": {}} for h in R + D}
    assert abs(radiant_win_probability(empty, R, D) - 0.5) < 1e-9

    strong = {h: {"vs": {}, "with": {}} for h in R + D}
    for r in R:
        strong[r]["vs"] = {d: (600, 1000) for d in D}
    p = radiant_win_probability(strong, R, D)
    assert 0.55 < p < 0.65, p

    # symmetric: dire's view of the same draft gives 1 - p
    mirror = {h: {"vs": {}, "with": {}} for h in R + D}
    for r in R:
        mirror[r]["vs"] = {d: (600, 1000) for d in D}
    for d in D:
        mirror[d]["vs"] = {r: (400, 1000) for r in R}
    assert abs(radiant_win_probability(mirror, D, R) - (1 - p)) < 1e-3
    print("ok")
