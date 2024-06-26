import os
from dotenv import load_dotenv

from src.datasetgenerator import StratzQuery, DatasetGenerator

def debug():
    load_dotenv()
    #querier = StratzQuery(os.getenv('STRATZ_TOKEN'))
    #querier.get_match(7590822094)

    csv_path = os.getenv('CSV_PATH')

    generator = DatasetGenerator(os.getenv('STRATZ_TOKEN'), csv_path)
    league = generator.add_professional_league(16842)

    print(len(league['matches']))

if __name__ == "__main__":
    debug()