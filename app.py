    with tabs[2]:
        if "team_a" in st.session_state and "team_b" in st.session_state:
            team_a_id = st.session_state["team_a"]["team"]["id"]
            team_b_id = st.session_state["team_b"]["team"]["id"]
            # Verifica se season_a e season_b estão no session_state, caso contrário usa um valor padrão
            season_a = st.session_state.get("season_a", 2025)
            season_b = st.session_state.get("season_b", 2025)
            games_a = get_team_games(team_a_id, season_a, home=True)
            games_b = get_team_games(team_b_id, season_b, home=False)
            if games_a:
                st.write(f"Jogos do Time A ({st.session_state['team_a']['team']['name']} - Mandante):")
                for game in games_a:
                    game_date = datetime.strptime(game["fixture"]["date"], "%Y-%m-%dT%H:%M:%S+00:00")
                    formatted_date = game_date.strftime("%d/%m/%Y %H:%M")
                    home_team = game["teams"]["home"]["name"]
                    away_team = game["teams"]["away"]["name"]
                    home_goals = game["goals"]["home"] if game["goals"]["home"] is not None else 0
                    away_goals = game["goals"]["away"] if game["goals"]["away"] is not None else 0
                    league_name = game["league"]["name"]
                    stats = get_game_stats(game["fixture"]["id"])
                    has_stats = bool(stats and any(stat["team"]["id"] == team_a_id for stat in stats))
                    title_suffix = " (SEM ESTATÍSTICAS)" if not has_stats else ""
                    title = f"{home_team} {home_goals} x {away_goals} {away_team} - {formatted_date} ({league_name}){title_suffix}"
                    with st.expander(title):
                        if has_stats:
                            data = []
                            for stat in stats:
                                if stat["team"]["id"] == team_a_id:
                                    for s in stat["statistics"]:
                                        value = s["value"]
                                        if s["type"].lower() == "ball possession" and isinstance(value, str):
                                            value = value.replace("%", "").strip()
                                            value = float(value) if value else 0.0
                                        else:
                                            try:
                                                value = float(value) if value is not None else 0.0
                                            except (ValueError, TypeError):
                                                value = 0.0
                                        data.append([s["type"], value])
                            df_stats = pd.DataFrame(data, columns=["Estatística", "Valor"])
                            df_stats["Valor"] = df_stats["Valor"].round(1)
                            st.dataframe(df_stats)
                        else:
                            st.write("Nenhuma estatística disponível para este jogo.")
            else:
                st.warning(f"Nenhum jogo finalizado encontrado para {st.session_state['team_a']['team']['name']} na temporada {season_a}. Tente outra temporada, como 2024.")
            if games_b:
                st.write(f"Jogos do Time B ({st.session_state['team_b']['team']['name']} - Visitante):")
                for game in games_b:
                    game_date = datetime.strptime(game["fixture"]["date"], "%Y-%m-%dT%H:%M:%S+00:00")
                    formatted_date = game_date.strftime("%d/%m/%Y %H:%M")
                    home_team = game["teams"]["home"]["name"]
                    away_team = game["teams"]["away"]["name"]
                    home_goals = game["goals"]["home"] if game["goals"]["home"] is not None else 0
                    away_goals = game["goals"]["away"] if game["goals"]["away"] is not None else 0
                    league_name = game["league"]["name"]
                    stats = get_game_stats(game["fixture"]["id"])
                    has_stats = bool(stats and any(stat["team"]["id"] == team_b_id for stat in stats))
                    title_suffix = " (SEM ESTATÍSTICAS)" if not has_stats else ""
                    title = f"{home_team} {home_goals} x {away_goals} {away_team} - {formatted_date} ({league_name}){title_suffix}"
                    with st.expander(title):
                        if has_stats:
                            data = []
                            for stat in stats:
                                if stat["team"]["id"] == team_b_id:
                                    for s in stat["statistics"]:
                                        value = s["value"]
                                        if s["type"].lower() == "ball possession" and isinstance(value, str):
                                            value = value.replace("%", "").strip()
                                            value = float(value) if value else 0.0
                                        else:
                                            try:
                                                value = float(value) if value is not None else 0.0
                                            except (ValueError, TypeError):
                                                value = 0.0
                                        data.append([s["type"], value])
                            df_stats = pd.DataFrame(data, columns=["Estatística", "Valor"])
                            df_stats["Valor"] = df_stats["Valor"].round(1)
                            st.dataframe(df_stats)
                        else:
                            st.write("Nenhuma estatística disponível para este jogo.")
            else:
                st.warning(f"Nenhum jogo finalizado encontrado para {st.session_state['team_b']['team']['name']} na temporada {season_b}. Tente outra temporada, como 2024.")
        else:
            st.info("Selecione os times na aba 'Seleção de Times' para ver os jogos.")
