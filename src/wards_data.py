from collections import defaultdict
from .datasetgenerator import DatasetGenerator
import pandas as pd

def get_match_wards(match:dict) -> dict:
    player_team_by_slot = {player['playerSlot']: player['isRadiant'] for player in match['players']}
    match_id = match['id']
    didRadiantWin = match['didRadiantWin']
    ward_events = match['playbackData']['wardEvents']
    wards = defaultdict(lambda: {})
    for w_e in ward_events:
        if w_e["action"] == "SPAWN":
            wards[w_e['indexId']] = {
                "id": f"{match_id}_{w_e['indexId']}",
                "match": match_id,
                "spawned_time": w_e["time"],
                "despawned_time": None,
                "positionX": w_e["positionX"],
                "positionY": w_e["positionY"],
                "wardType": w_e["wardType"],
                "isRadiant": player_team_by_slot[w_e["fromPlayer"]],
                "playerDestroyed": w_e["playerDestroyed"],
                "didRadiantWin": didRadiantWin,
            }
            continue
        wards[w_e['indexId']]["despawned_time"] = w_e['time']
    

    # get kills around
    
    return dict(wards)

from collections import defaultdict

def map_death_count(df_wards, df_deaths):
    death_time = df_deaths['time'].item()
    return (df_wards['spawned_time'] < death_time) & (death_time > df_wards['despawned_time'])

def get_df_match_wards(match:dict) -> dict:
    if not match['playbackData']:
        print(f"warning: no playbackData found for match {match['id']}")
        return
    wards = get_match_wards(match)
    if len(wards) == 0:
        print(f"warning: no wards found for match {match['id']}")
        return
    df_wards = defaultdict(lambda: [])
    for id, w in wards.items():
        for key, item in w.items():
            df_wards[key].append(item)
    df_wards = pd.DataFrame.from_dict(df_wards)

    none_despawned_mask = df_wards['despawned_time'].isna()
    df_wards.loc[none_despawned_mask, 'despawned_time'] = match['durationSeconds']

    # df_deaths = get_deaths_through_ward(match)
    
    # df_wards['possible_enemies_death'] = 0
    # return map_death_count(df_wards, df_deaths)

    
    # df_wards['possible_enemies_death'] = 0


    df_wards['spawned_time_minute'] = (df_wards['spawned_time'] // 60).astype(int)
    df_wards['despawned_time_minute'] = (df_wards['despawned_time'] // 60).astype(int)
    
    df_wards['radiantTeam'] = match['radiantTeam']['name']
    df_wards['direTeam'] = match['direTeam']['name']


    return df_wards

def get_league_df_wards(league: dict):
    matches = league['matches']
    ward_matches_df = [get_df_match_wards(match) for match in matches]
    df_wards = pd.concat(ward_matches_df)
    df_wards['league'] = league['id']
    df_wards['region'] = league['region']
    return df_wards

def get_leagues_df_wards(generator: DatasetGenerator, leagues_ids: dict):
    leagues = generator.get_professional_league(leagues_ids)
    df_wards = []
    for league_id, league in leagues.items():
        matches = league['matches']
        ward_matches_df = [get_df_match_wards(match) for match in matches]
        ward_matches_df = pd.concat(ward_matches_df)
        ward_matches_df['league'] = league_id
        ward_matches_df['region'] = league['region']
        ward_matches_df['leagueName'] = league['displayName']
        df_wards.append(ward_matches_df)
    return pd.concat(df_wards)

def merge_by_time(df):
    def create_time_rows(row):
        return [{'time': t, **row} for t in range(row['spawned_time_minute'], row['despawned_time_minute']) if row['id']]
    # Apply the function and expand the DataFrame
    df_expanded = df.apply(create_time_rows, axis=1).explode().reset_index(drop=True)

    # Convert the expanded rows from dictionaries back to a DataFrame
    df_expanded = pd.DataFrame([row for row in df_expanded.tolist() if isinstance(row, dict)])

    # Drop the original columns (optional)
    df_expanded.drop(['spawned_time', 'despawned_time','spawned_time_minute', 'despawned_time_minute'], axis=1, inplace=True)

    # Print the expanded DataFrame
    return df_expanded


class WardDataset:

    POSITION_X_OFFSET = -50
    POSITION_Y_OFFSET = -54

    POSITION_RESCALER_FACTOR = 0.5

    def __init__(self, STRATZ_TOKEN: str):
        self.generator = DatasetGenerator(STRATZ_TOKEN)


    def get_by_time_merged_dataset(self):
        df = get_leagues_df_wards(self.generator, [16842, 16840, 16841, 16839, 16843, 16844, 16776, 16777, 16779, 16778, 16774, 16775])
        df = merge_by_time(df)
        df['positionX'] = ((df['positionX'] + WardDataset.POSITION_X_OFFSET) * WardDataset.POSITION_RESCALER_FACTOR).astype(int)
        df['positionY'] = ((df['positionY'] + WardDataset.POSITION_Y_OFFSET) * WardDataset.POSITION_RESCALER_FACTOR).astype(int)
        df['didWardWin'] = df['didRadiantWin'] == df['isRadiant']
        return df