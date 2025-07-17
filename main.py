# main.py - Hauptagent und Programmsteuerung

import os
from typing import List, Tuple, Dict, Any
import asyncio
import time
import logging

from dotenv import load_dotenv
from datetime import datetime

from pydantic_ai import Agent, RunContext, Tool, exceptions
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

import aiohttp
import ssl
import certifi

from logger_config import setup_logging

setup_logging()

# Importieren der spezialisierten Agenten
from agents import *

# Zum Testen der Fragen
import pandas as pd
from tqdm import tqdm

# Logging-Konfiguration

from functools import wraps

logger = logging.getLogger("main")

load_dotenv()
LLM_NAME = os.getenv("LLM_NAME")
LLM_URL = os.getenv("LLM_URL")
LLM_PORT = os.getenv("LLM_PORT")


# Logging-Decorator
def log_tool(func):
    """Wraps a tool function to log when it's called."""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger.info(f"Tool called: {func.__name__}")
        return await func(*args, **kwargs)

    return async_wrapper


logger.info("Connect to the LLM...")
model = OpenAIModel(
    LLM_NAME,
    provider=OpenAIProvider(
        base_url=f'http://{LLM_URL}:{LLM_PORT}/v1', api_key='your-api-key'
    ),
)
logger.info("Connection successfully.")

# Hauptagent
agent = Agent(
    model,
    deps_type=str,
    system_prompt=(
        f"""## 🤖 **System Prompt für OTH-AW AI Master-Agenten**

Du bist der zentrale AI-Chatbot der Ostbayerischen Technischen Hochschule Amberg-Weiden (**OTH-AW**) 🎓.  
Deine Aufgabe ist es, **Fragen entgegenzunehmen und die passenden spezialisierten Unteragenten** damit zu beauftragen, die beste Antwort zu liefern 📤➡️📥.  
Du unterstützt Studierende, Lehrende und Mitarbeitende mit **schnellen, verlässlichen und strukturierten Antworten** – gerne mit vielen passenden Emojis 😄✨.

---

### 🎯 **Deine Rolle**

- Du bist der **Master-Agent**, der die Übersicht über alle Themengebiete hat.
- Deine Aufgabe ist es **nicht**, selbst Inhalte zu beantworten – du **leitest Anfragen gezielt an spezialisierte Agenten** weiter.
- Du entscheidest anhand der Frage, **welcher deiner 9 Unteragenten** zuständig ist (z. B. "Bibliothek", "IT-Support", "Studiengänge" etc.).

---

### 🧭 **Was du tust**

1. **Kategorisierung**: Analysiere die Nutzerfrage und bestimme, welche Kategorie (Agent) zuständig ist.
2. **Delegation**: Rufe das passende Tool/Funktion auf, um die Frage zu delegieren (z. B. `deligate_to_it_support_agent(...)`).
3. **Antwortaufbereitung**: Du gibst die Antwort des spezialisierten Agenten formatiert an den Nutzer zurück – **freundlich, strukturiert, mit Emojis**.

---

### 🧩 **Unteragenten (Beispiele)**

- `search_research_agent`: Websuche, FAQ, Informationsbeschaffung 🔍  
- `study_programs_agent`: Studiengänge, Module, Prüfungen 📚  
- `library_agent`: Öffnungszeiten, Kataloge, Datenbanken 📖  
- `it_support_agent`: WLAN, VPN, Remote-Zugang, Tools 💻  
- u.v.m. – **du kennst alle Kategorien und beschreibst sie exakt**.

---

### 💬 **Verhaltensregeln**

- Sprich **Deutsch oder Englisch**, je nach Eingabe der Nutzer:innen 🌍  
- Antworte wie im Chat: **kurze, strukturierte Absätze**, gerne mit Emojis  
- Wenn unklar, bitte freundlich um Präzisierung  
- Verweise bei Unsicherheiten an **offizielle OTH-AW-Stellen oder Webseiten**  
- Bleibe **neutral, professionell, hilfreich** – keine Meinungen oder Spekulationen

---

### 🗓️ **Aktuelles Datum**: {datetime.today().strftime('%Y-%m-%d')}

---
‼️ **Wichtig:** Du bist nicht irgendein Chatbot – du bist der zentrale Master-Agent der OTH-AW.  
Die Qualität deiner Delegation beeinflusst das gesamte System! 🎯

‼️ Falls du gefragt wirst was deine Aufgabe  ist, dann antwortest du, dass du der zentrale AI-Chatbot der Ostbayerischen 
Technischen Hochschule Amberg-Weiden (**OTH-AW**) 🎓 bist und dich freust die Fragen zu beantworten. ‼️

"""

    )
)


# Agents

# -------------------
# 1) Suche & Recherche - Agent
# -------------------
@agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def delegate_to_search_research_agent(ctx: RunContext, question: str) -> str:
    """
    Dieser Agent ist spezialisiert auf allgemeine und gezielte Informationsbeschaffung. Er nutzt Websuche,
    interne FAQ-Datenbanken und gezielten Webseitenzugriff, um Antworten auf vielseitige Fragen zu liefern. Tools:
    websearch, get_top_n_similar_questions_from_FAQ, get_specific_website

    Args:
        ctx (RunContext): Kontextobjekt, das Metainformationen über die Ausführung enthält – z. B. Session-Status oder Benutzerinformationen.
        question (str): Die vom Benutzer gestellte Frage, die durch diesen Agenten beantwortet werden soll.

    Returns:
        str: Die Antwort auf die gestellte Frage, erzeugt durch den spezialisierten Agenten.
    """
    track_tool()
    logger.info("Delegating question to the search_research_agent...")
    try:
        result = await search_research_agent.run(
            user_prompt=question
        )
        if isinstance(result.output, str):
            output = result.output.split("</think>")[-1] if "</think>" in result.output else result.output
        else:
            output = str(result.output)
        logger.debug(f"Response from the search_research_agent: {output}")
        return output
    except exceptions.UsageLimitExceeded:
        logger.error("Usage limit exceeded for search_research_agent")
        return "Entschuldigung, ich habe derzeit mein Anfragelimit erreicht. Bitte versuchen Sie es später erneut."
    except Exception as e:
        logger.error(f"Error with the search_research_agent... {e}", exc_info=True)
        return f"Error occurred while delegating to search_research_agent: {str(e)}"


# -------------------
# 2) Studiengänge & Studium allgemein - Agent
# -------------------
@agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def deligate_to_study_programs_studies_in_general_agent(ctx: RunContext, question: str) -> str:
    """
    Dieser Agent deckt alle organisatorischen und fachlichen Fragen rund ums Studium ab – von Modulplänen über
    Fakultätsinfos bis hin zu Themen wie Rückmeldung, Beurlaubung oder Studiengangwechsel, Stundenpläne. Stundenplan.
    Tools: get_study_program_overview, get_specific_study_program, get_overview_of_one_faculty,
    get_specific_study_schedule_stundenplan, get_specific_study_examination_plans_pruefungsplan,
    get_information_change_study_programm_study_other_topic,
    get_information_on_taking_a_leave_of_absence_during_study, get_information_exmatriculation,
    get_information_for_students_re_registration

    Args:
        ctx (RunContext): Kontextobjekt, das Informationen zur Laufzeit bereitstellt – z. B. Benutzer-Session, Metadaten oder andere Ausführungsdetails.
        question (str): Die konkrete Frage des Benutzers zu Studiengängen oder organisatorischen Aspekten des Studiums.

    Returns:
        str: Eine vom spezialisierten Agenten generierte Antwort auf die Benutzeranfrage.
    """
    track_tool()
    logger.info("Delegating question to the study_programs_studies_in_general_agent...")
    try:
        result = await study_programs_studies_in_general_agent.run(user_prompt=question)
        output = result.output.split("</think>")[-1] if "</think>" in result.output else result.output
        logger.debug(f"Response from the study_programs_studies_in_general_agent: {output}")
        return output
    except Exception as e:
        logger.error(f"Error with the study_programs_studies_in_general_agent... {e}", exc_info=True)
        raise


# -------------------
# 3) Formulare & Organisatorische Dokumente - Agent
# -------------------
@agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def deligate_to_forms_documents_agent(ctx: RunContext, question: str) -> str:
    """
    Dieser Agent liefert gezielt Formulare und Richtlinien – etwa für Praxissemester oder Abschlussarbeiten – und
    erklärt deren Inhalte sowie Anwendungskontexte. Tools: get_internship_semester_praxissemester_documents,
    get_forms_guidelines_bachelor_master_thesis_documents

    Args:
        ctx (RunContext): Kontextobjekt, das Informationen zur Laufzeit bereitstellt – z. B. Benutzer-Session, Metadaten oder andere Ausführungsdetails.
        question (str): Die konkrete Frage des Benutzers zu Studiengängen oder organisatorischen Aspekten des Studiums.

    Returns:
        str: Eine vom spezialisierten Agenten generierte Antwort auf die Benutzeranfrage.
    """
    track_tool()
    logger.info("Delegating question to the forms_documents_agent...")
    try:
        result = await forms_documents_agent.run(user_prompt=question)
        output = result.output.split("</think>")[-1] if "</think>" in result.output else result.output
        logger.debug(f"Response from the forms_documents_agent: {output}")
        return output
    except Exception as e:
        logger.error(f"Error with the forms_documents_agent... {e}", exc_info=True)
        raise


# -------------------
# 4) Personal & Kontakte - Agent
# -------------------
@agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def deligate_to_personnel_contacts_agent(ctx: RunContext, question: str) -> str:
    """
    Dieser Agent liefert strukturierte Informationen zu Mitarbeitenden und Lehrenden der Hochschule – inklusive
    Kontaktdaten, Zuständigkeiten und Profilinformationen. Tools: get_employee_professor_info, get_details_about_prof

    Args:
        ctx (RunContext): Kontextobjekt, das Informationen zur Laufzeit bereitstellt – z. B. Benutzer-Session, Metadaten oder andere Ausführungsdetails.
        question (str): Die konkrete Frage des Benutzers zu Studiengängen oder organisatorischen Aspekten des Studiums.

    Returns:
        str: Eine vom spezialisierten Agenten generierte Antwort auf die Benutzeranfrage.
    """
    track_tool()
    logger.info("Delegating question to the personnel_contacts_agent...")
    try:
        result = await personnel_contacts_agent.run(user_prompt=question)
        output = result.output.split("</think>")[-1] if "</think>" in result.output else result.output
        logger.debug(f"Response from the personnel_contacts_agent: {output}")
        return output
    except Exception as e:
        logger.error(f"Error with the personnel_contacts_agent... {e}", exc_info=True)
        raise


# -------------------
# 5) Hochschule & Campusservices - Agent
# -------------------
@agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def deligate_to_campus_services_agent(ctx: RunContext, question: str) -> str:
    """
    Dieser Agent deckt alle digitalen und physischen Services der Hochschule ab – von WLAN-Zugang über Kursanmeldung
    bis hin zu Lageplänen, Kalendern und Campus-Apps. Tools: kursanmeldung_link, primuss_campus_link,
    get_wifi_eduroam_vpn_info, get_email_contact_calender, get_info_for_file_exchange,
    get_info_for_printing_on_the_oth, get_site_map_of_different_location_amberg_weiden,
    get_Vorlesungs_Pruefungs_und_vorlesungsfreie_Zeiten, get_OTH_AW_app_information, get_terminplaner_link,
    get_gitlab_link. Für stundenpläne oder ähnliches similar sei auf "deligate_to_study_programs_studies_in_general_agent" verwiesen.

    Args:
        ctx (RunContext): Kontextobjekt, das Informationen zur Laufzeit bereitstellt – z. B. Benutzer-Session, Metadaten oder andere Ausführungsdetails.
        question (str): Die konkrete Frage des Benutzers zu Studiengängen oder organisatorischen Aspekten des Studiums.

    Returns:
        str: Eine vom spezialisierten Agenten generierte Antwort auf die Benutzeranfrage.
    """
    track_tool()
    logger.info("Delegating question to the campus_services_agent...")
    try:
        result = await campus_services_agent.run(user_prompt=question)
        output = result.output.split("</think>")[-1] if "</think>" in result.output else result.output
        logger.debug(f"Response from the campus_services_agent: {output}")
        return output
    except Exception as e:
        logger.error(f"Error with the campus_services_agent... {e}", exc_info=True)
        raise


# -------------------
# 6) Bibliothek - Agent
# -------------------
@agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def deligate_to_library_agent(ctx: RunContext, question: str) -> str:
    """
    Dieser Agent ist auf bibliotheksbezogene Informationen spezialisiert – Öffnungszeiten, Kontakt, Online-Kataloge,
    digitale Ressourcen, Leihsysteme und Veranstaltungen. Tools: get_contact_data_of_library,
    get_current_news_from_library_relevant_for_students, get_current_news_from_library_specific,
    get_current_news_events_from_library_more_informations, get_library_opening_hours, get_library_24h_opening_info,
    get_library_team_contact_data, get_library_note_as_external_user,
    get_library_OPAC_Online_Public_Access_Catalogue_info, get_digital_library_info, get_library_list_of_databases,
    get_library_access_to_journals_info, get_library_lend_info, get_general_question_for_library

    Args:
        ctx (RunContext): Kontextobjekt, das Informationen zur Laufzeit bereitstellt – z. B. Benutzer-Session, Metadaten oder andere Ausführungsdetails.
        question (str): Die konkrete Frage des Benutzers zu Studiengängen oder organisatorischen Aspekten des Studiums.

    Returns:
        str: Eine vom spezialisierten Agenten generierte Antwort auf die Benutzeranfrage.
    """
    track_tool()
    logger.info("Delegating question to the library_agent...")
    try:
        result = await library_agent.run(user_prompt=question)
        output = result.output.split("</think>")[-1] if "</think>" in result.output else result.output
        logger.debug(f"Response from the library_agent: {output}")
        return output
    except Exception as e:
        logger.error(f"Error with the library_agent... {e}", exc_info=True)
        raise


# -------------------
# 7) Wissenschaftliches Arbeiten & Zitieren - Agent
# -------------------
@agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def deligate_to_academic_writing_citation_agent(ctx: RunContext, question: str) -> str:
    """
    Dieser Agent unterstützt beim wissenschaftlichen Arbeiten – insbesondere beim Einsatz von Literaturverwaltungs-
    und Zitier-Tools wie Citavi oder Alternativen. Tools: get_citation_tool_info_citavi,
    get_citation_tool_info_alternatives

    Args:
        ctx (RunContext): Kontextobjekt, das Informationen zur Laufzeit bereitstellt – z. B. Benutzer-Session, Metadaten oder andere Ausführungsdetails.
        question (str): Die konkrete Frage des Benutzers zu Studiengängen oder organisatorischen Aspekten des Studiums.

    Returns:
        str: Eine vom spezialisierten Agenten generierte Antwort auf die Benutzeranfrage.
    """
    track_tool()
    logger.info("Delegating question to the academic_writing_citation_agent...")
    try:
        result = await academic_work_citation_agent.run(user_prompt=question)
        output = result.output.split("</think>")[-1] if "</think>" in result.output else result.output
        logger.debug(f"Response from the academic_writing_citation_agent: {output}")
        return output
    except Exception as e:
        logger.error(f"Error with the academic_writing_citation_agent... {e}", exc_info=True)
        raise


# -------------------
# 8) Rechenzentrum / IT-Support - Agent
# -------------------
@agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def deligate_to_it_support_agent(ctx: RunContext, question: str) -> str:
    """
    Dieser Agent liefert aktuelle IT-Informationen, Hilfe zu technischen Problemen, FAQs und Unterstützung beim
    Remote-Zugriff auf Hochschulressourcen. Tools: get_rechenzentrum_data_center_news_for_students,
    get_rechenzentrum_FAQ, get_help_for_remote_access_labor_pc

    Args:
        ctx (RunContext): Kontextobjekt, das Informationen zur Laufzeit bereitstellt – z. B. Benutzer-Session, Metadaten oder andere Ausführungsdetails.
        question (str): Die konkrete Frage des Benutzers zu Studiengängen oder organisatorischen Aspekten des Studiums.

    Returns:
        str: Eine vom spezialisierten Agenten generierte Antwort auf die Benutzeranfrage.
    """
    track_tool()
    logger.info("Delegating question to the it_support_agent...")
    try:
        result = await it_support_agent.run(user_prompt=question)
        output = result.output.split("</think>")[-1] if "</think>" in result.output else result.output
        logger.debug(f"Response from the it_support_agent: {output}")
        return output
    except Exception as e:
        logger.error(f"Error with the it_support_agent... {e}", exc_info=True)
        raise


# -------------------
# 9) Unterstützung & Services - Agent
# -------------------
@agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def deligate_to_support_services_agent(ctx: RunContext, question: str) -> str:
    """
    Dieser Agent informiert über hochschulweite Unterstützungsangebote: Kompetenzzentren, fachspezifische
    Einrichtungen, Lehrunterstützung, Videoerstellung und Schwarzes Brett. Tools: get_help_for_teaching,
    get_help_for_creating_video, get_kompetenzzentrum_informationen, get_specialized_institution_information,
    get_schwarzes_Brett_infos_fuer_studenten

    Args:
        ctx (RunContext): Kontextobjekt, das Informationen zur Laufzeit bereitstellt – z. B. Benutzer-Session, Metadaten oder andere Ausführungsdetails.
        question (str): Die konkrete Frage des Benutzers zu Studiengängen oder organisatorischen Aspekten des Studiums.

    Returns:
        str: Eine vom spezialisierten Agenten generierte Antwort auf die Benutzeranfrage.
    """
    track_tool()
    logger.info("Delegating question to the support_services_agent...")
    try:
        result = await support_services_agent.run(user_prompt=question)
        output = result.output.split("</think>")[-1] if "</think>" in result.output else result.output if "</think>" in result.output else result.output
        logger.debug(f"Response from the support_services_agent: {output}")
        return output
    except Exception as e:
        logger.error(f"Error with the support_services_agent... {e}", exc_info=True)
        raise


# -------------------
# -------------------

def test_fragen(file: str = 'test_analysis/fragen_10_04_2025.csv'):
    outer_questions = pd.read_csv(file)
    results = []
    for frage in tqdm(outer_questions["Bitte formuliere hier deine Frage:"], unit="Frage(n)",
                      desc="Beantworte Fragen..."):
        start_time = time.perf_counter()
        anfrage = agent.run_sync(user_prompt=frage)
        end_time = time.perf_counter()
        results.append({"question": frage, "answer": str(anfrage.data),#.split("</think>")[1]),
                        "zeitdauer": f"{end_time - start_time:.6f}"})

    with open("test_analysis/answers.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

# Was gibt es aktuelles auf dem schwarzen Brett?
# Elapsed time: 116.045001 seconds QwQ 32B


async def main():

    # Die Zertifikate gehen manchmal nicht -> Deshalb diese Bibliothek (certifi)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    async with aiohttp.ClientSession(connector=connector) as session:
        user_input = "None"
        res = None  # initialisieren, um später auf res.new_messages() zugreifen zu können
        while user_input != "exit":
            user_input = input("Your Question: ")
            start_time = time.perf_counter()
            try:
                if res:
                    run_result = await agent.run(
                        user_prompt=user_input,
                        message_history=res.new_messages()
                    )
                else:
                    run_result = await agent.run(user_prompt=user_input)
            except exceptions.UsageLimitExceeded:
                print("Limit Exceeded")
                # print(user_input)
                continue
            except Exception as e:
                print(f"Error occurred: {e}")
                continue

            # print(run_result.output)
            result_text = await check_valid_links(run_result.output, session)#.split("</think>")[1])
            print(result_text)
            print("\n\n----------------------------\n\n")
            end_time = time.perf_counter()
            print(f"Elapsed time: {end_time - start_time:.6f} seconds")
            print("\n\n----------------------------\n\n")
            # print(run_result.new_messages_json())

            res = run_result


if __name__ == "__main__":
    asyncio.run(main())
