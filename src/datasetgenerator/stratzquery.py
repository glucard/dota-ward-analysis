import json, requests

class StratzQuery:

    stratz_graphql_url = "https://api.stratz.com/graphql"

    def __init__(self, token):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def graphql(self, graphql_query):
        data = {
            "query": graphql_query
        }
        json_data = json.dumps(data)
        response = requests.post(url=self.stratz_graphql_url, headers=self.headers, data=json_data)
        response.raise_for_status()
        result = response.json()
        return result
    
    def get_professional_league(self, league_id):
        graphql_query = f"""
        {{
            league (id: {league_id}) {{
                id
                name
                banner
                tier
                region
                startDateTime
                endDateTime
                tournamentUrl
                lastMatchDate
                displayName
                description
                matches (request: {{
                    skip: 0,
                    take: 100
                    }}) {{
                    id
                    didRadiantWin
                    durationSeconds
                    startDateTime
                    firstBloodTime
                    averageRank
                    players {{
                        steamAccountId
                        steamAccount {{
                        rankShift
                        isAnonymous
                        isStratzPublic
                        seasonRank
                        }}
                        hero {{
                        displayName
                        }}
                    }}
                    playbackData {{
                        wardEvents {{
                        indexId
                        time
                        positionX
                        positionY
                        fromPlayer
                        wardType
                        action
                        playerDestroyed
                        }}
                        towerDeathEvents {{
                        time
                        radiant
                        dire
                        }}
                    }}
                    averageImp
                    actualRank
                    averageRank
                    radiantKills
                    direKills
                    radiantTeam {{
                        id
                        name
                        countryName
                    }}
                    direTeam {{
                        id
                        name
                        countryName
                    }}
                    }}
                }}
            }}

        """
        return self.graphql(graphql_query)