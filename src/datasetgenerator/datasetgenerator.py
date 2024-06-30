import os, json
import pandas as pd
from .stratzquery import StratzQuery


def flatten_json(b, delim):
    val = {}
    for i in b.keys():
        if isinstance(b[i], dict):
            get = flatten_json(b[i], delim)
            for j in get.keys():
                val[i + delim + j] = get[j]
        elif isinstance(b[i], list):
            for j, e in enumerate(b[i]):
                if isinstance(e, dict):
                    get = flatten_json(e, delim)
                    for k in get.keys():
                        val[f"{i}{delim}{j}{delim}{k}"] = get[k]
                else:
                    val[f"{i}{delim}{j}"] = j

        else:
            val[i] = b[i]
            
    return val

def check_temp_folder():
    if "temp" not in os.listdir():
        os.mkdir('temp')

def filter_match(match_data):
    return match_data

class DatasetGenerator:
    def __init__(self, stratz_token):
        self.querier = StratzQuery(stratz_token)
        check_temp_folder()


    def get_match_by_id(self, id):
        match = self.querier.get_match(id)['data']
        try:
            match['didRadiantWin']
        except:
            raise ValueError("Could not fetch match")
        return match

    def add_match(self, match_data):
        flattened_match = flatten_json(match_data, "_")
        match_csv_format = {}
        for k, e in flattened_match.items():
            match_csv_format[k] = [e]
        
        match_df = pd.DataFrame.from_dict(match_csv_format)
        self.csv_add_match(match_df)

    def add_match_by_id(self, id):
        match_data = self.get_match_by_id(id)
        self.add_match(match_data)

    def add_matches(self, matches_data):
        matches_df_format = {}
        flattened_matches = [flatten_json(match, "_") for match in matches_data]
        #n = len(flattened_matches)
        for i, match in enumerate(flattened_matches):
            #print(f"{i+1}/{n}")
            match_keys = match.keys()
            df_match_keys = matches_df_format.keys()

            match_ausent_keys = [key for key in match_keys if key not in df_match_keys]
            df_ausent_keys = [key for key in df_match_keys if key not in match_keys]

            for key in match_ausent_keys:
                matches_df_format[key] = [None for _ in range(i)]
            for key in df_ausent_keys:
                matches_df_format[key].append(None)
            for key, element in match.items():
                matches_df_format[key].append(element)
        df = pd.DataFrame.from_dict(matches_df_format)
        self.csv_add_match(df)
        
    def get_professional_league(self, league_ids: list, update_leagues_json=False) -> dict:

        leagues_path = f"temp{os.sep}leagues"
        if not os.path.isdir(leagues_path):
            os.mkdir(leagues_path)

        # find ausent json leagues
        json_leagues_ids = os.listdir(leagues_path)
        not_in_json_leagues = [id for id in league_ids if f"{id}.json" not in json_leagues_ids]

        # if update leagues, then treat all like its not have a json 
        if update_leagues_json:
            not_in_json_leagues = league_ids

        # get ausent json leagues
        leagues = {}
        for to_dl_id in not_in_json_leagues:
            d_l = self.querier.get_professional_league(to_dl_id)['data']['leagues'][0]
            with open(f'{leagues_path}{os.sep}{d_l["id"]}.json', 'w') as f:
                json.dump(d_l, f)

        # load leagues from json
        leagues = {}
        for league_id in league_ids:
            with open(f'{leagues_path}{os.sep}{league_id}.json', 'r') as f:
                league = json.loads(f.read())
            leagues[league_id] = league

        # matches = []
        # for match in league['matches']:
        #     matches.append(match)
        #     #self.add_match(match)
    
        # matches = [filter_match(match) for match in matches]
        # self.add_matches(matches)

        return leagues