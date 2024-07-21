# pip install statsbombpy
from statsbombpy import sb

competitions = sb.competitions()
matches = sb.matches(competition_id=55, season_id=282)
lineups = sb.lineups(match_id=3943043)