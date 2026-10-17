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

RECOGNIZE_PROMPT = """This is a Dota 2 screenshot with two teams of heroes (Radiant and Dire).
Identify all 10 heroes. Radiant is the left/top/green team, Dire is the right/bottom/red team.
Answer with JSON only: {"radiant": [5 hero names], "dire": [5 hero names]}.
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
        r.raise_for_status()
        data = r.json()
    if data.get("errors"):
        raise RuntimeError(f"STRATZ: {data['errors'][0]['message']}")
    return data["data"]


async def get_heroes():
    global _heroes
    if _heroes is None:
        data = await _stratz("{ constants { heroes { id displayName } } }")
        _heroes = {h["displayName"]: h["id"] for h in data["constants"]["heroes"]}
    return _heroes


async def recognize_teams(image_bytes):
    heroes = await get_heroes()
    image_b64 = base64.b64encode(image_bytes).decode()
    r = await openai_utils.azureclient_swc.responses.create(
        model=VISION_DEPLOYMENT,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": RECOGNIZE_PROMPT + ", ".join(sorted(heroes)),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                    },
                ],
            }
        ],
        reasoning={"effort": "low"},
        text={"format": {"type": "json_object"}},
        timeout=90.0,
    )
    teams = json.loads(r.output_text)
    return teams["radiant"], teams["dire"], (r.usage.input_tokens, r.usage.output_tokens)


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
        "{ heroStats { "
        + " ".join(
            f"h{h}: matchUp(heroId: {h}, bracketBasicIds: [{BRACKET}], take: 200) "
            f"{{ vs {{ {fields} }} with {{ {fields} }} }}"
            for h in hero_ids
        )
        + " } }"
    )
    data = (await _stratz(query))["heroStats"]
    stats = {}
    for h in hero_ids:
        dryad = (data[f"h{h}"] or [{}])[0]
        stats[h] = {
            kind: {
                row["heroId2"]: (row["winCount"] or 0, row["matchCount"] or 0)
                for row in dryad.get(kind) or []
            }
            for kind in ("vs", "with")
        }
    return stats


async def analyze_draft(image_bytes):
    """Returns (html answer, (n_input_tokens, n_output_tokens))."""
    radiant, dire, usage = await recognize_teams(image_bytes)
    heroes = await get_heroes()
    names = lambda team: html.escape(", ".join(map(str, team)))

    unknown = [h for h in radiant + dire if h not in heroes]
    if len(radiant) != 5 or len(dire) != 5 or unknown:
        text = "🥲 Не удалось распознать 5 на 5 героев.\n"
        text += f"Radiant: {names(radiant)}\nDire: {names(dire)}"
        if unknown:
            text += f"\nНеизвестные: {names(unknown)}"
        return text, usage

    radiant_ids = [heroes[h] for h in radiant]
    dire_ids = [heroes[h] for h in dire]
    stats = await fetch_matchups(radiant_ids + dire_ids)
    p = radiant_win_probability(stats, radiant_ids, dire_ids)

    text = f"🟢 <b>Radiant:</b> {names(radiant)}\n"
    text += f"🔴 <b>Dire:</b> {names(dire)}\n\n"
    text += f"📊 Вероятность победы ({BRACKET}, STRATZ):\n"
    text += f"🟢 Radiant: <b>{p:.1%}</b>\n"
    text += f"🔴 Dire: <b>{1 - p:.1%}</b>"
    return text, usage


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
