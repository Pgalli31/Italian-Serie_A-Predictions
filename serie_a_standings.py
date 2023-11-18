import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Creating link and requesting download using requests
standings_url = "https://fbref.com/en/comps/11/2022-2023/2022-2023-Serie-A-Stats#all_results2022-2023111"
data = requests.get(standings_url)

soup = BeautifulSoup(data.text)

# Getting the standings table from the site
standings_table = soup.select('table.stats_table')[0]

# Getting the squad links from the standings table
links = standings_table.find_all('a')
links = [l.get('href') for l in links]
links = [l for l in links if '/squads/' in l]

# Getting the team stats
team_urls = [f"https://fbref.com{l}" for l in links]
team_url = team_urls[0]
data = requests.get(team_url)

# Getting the Scores & Fixtures table
matches = pd.read_html(data.text, match="Scores & Fixtures")

# Getting the shooting stats
soup = BeautifulSoup(data.text)
links = soup.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if l and 'all_comps/shooting/' in l]

# Reading the shooting stats and dropping a level
data = requests.get(f"https://fbref.com{links[0]}")
shooting = pd.read_html(data.text, match="Shooting")[0]
shooting.columns = shooting.columns.droplevel()

# Merging the shooting stats
team_data = matches[0].merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")

# League Standing years we want
years = list(range(2023, 2021, -1))

all_matches = []

standings_url = "https://fbref.com/en/comps/11/2022-2023/2022-2023-Serie-A-Stats#all_results2022-2023111"

for year in years:
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text)
    standings_table = soup.select('table.stats_table')[0]

    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]

    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://fbref.com/{previous_season}"

    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")

        data = requests.get(team_url)
        matches = pd.read_html(data.text, match="Scores & Fixtures")[0]

        soup = BeautifulSoup(data.text)
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        data = requests.get(f"https://fbref.com{links[0]}")
        shooting = pd.read_html(data.text, match="Shooting")[0]
        shooting.columns = shooting.columns.droplevel()

        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue
        
        team_data = team_data[team_data["Comp"] == "Serie A"]
        team_data["Season"] = year
        team_data["Team"] = team_name
        all_matches.append(team_data)
        time.sleep(1)

match_df = pd.concat(all_matches)
match_df.to_csv("serie_a_matches.csv")
