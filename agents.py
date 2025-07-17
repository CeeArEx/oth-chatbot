import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple, Coroutine
import aiohttp
import asyncio

from pydantic_ai import Agent, RunContext, exceptions
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel
from functools import wraps

# Importiere die Tool-Implementierungen
from my_tools import *

logger = logging.getLogger("agent")

load_dotenv()

LLM_NAME = os.getenv("LLM_NAME")
LLM_URL = os.getenv("LLM_URL")
LLM_PORT = os.getenv("LLM_PORT")


def log_tool(func):
    """Wraps a tool function to log when it's called."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger.info(f"Tool called: {func.__name__}")
        return await func(*args, **kwargs)

    return wrapper


model = OpenAIModel(
    LLM_NAME,  # "QwQ_32B", "Qwen3-32B"
    provider=OpenAIProvider(
        base_url=f'http://{LLM_URL}:{LLM_PORT}/v1', api_key='your-api-key'
    ),
)

# -------------------
# 1) Suche & Recherche - Agent
# -------------------
search_research_agent = Agent(
    model,
    deps_type=str,
    system_prompt="Du bist verantwortlich für allgemeine und spezifische Informationsrecherche. Du findest Webseiten, "
                  "durchsuchst externe sowie interne Quellen und beantwortest Fragen basierend auf existierenden "
                  "FAQ-Einträgen oder Webseiteninhalten.",
    instrument=True,
)

# -------------------
# 2) Studiengänge & Studium allgemein - Agent
# -------------------
study_programs_studies_in_general_agent = Agent(
    model,
    deps_type=str,
    system_prompt=(
        "Du gibst strukturierte Auskünfte zu Studienprogrammen, Fakultäten, Stundenplänen, Prüfungsplänen und "
        "allgemeinen organisatorischen Abläufen im Studium – inklusive Rückmeldung, Exmatrikulation, "
        "Studiengangwechsel und Beurlaubung."
    ),
    instrument=True,
)

# -------------------
# 3) Formulare & Organisatorische Dokumente - Agent
# -------------------
forms_documents_agent = Agent(
    model,
    deps_type=str,
    system_prompt=(
        "Du bist zuständig für das Bereitstellen und Erklären von offiziellen Formularen und Dokumenten im Kontext "
        "von Praxissemester, Abschlussarbeiten und weiteren studienbezogenen Verwaltungsprozessen."
    ),
    instrument=True,
)

# -------------------
# 4) Personal & Kontakte - Agent
# -------------------
personnel_contacts_agent = Agent(
    model,
    deps_type=str,
    system_prompt=(
        "Du gibst Auskunft über Mitarbeiter:innen und Professor:innen – inklusive Kontaktinformationen und "
        "spezifischer Rollen oder Zuständigkeiten an der Hochschule."
    ),
    instrument=True,
)

# -------------------
# 5) Hochschule & Campusservices - Agent
# -------------------
campus_services_agent = Agent(
    model,
    deps_type=str,
    system_prompt=(
        "Du unterstützt bei Fragen zu hochschulweiten Services: Campusportale, WLAN/VPN, Kalender, Dateiaustausch, "
        "Drucken, Standortpläne, Fristen, Apps und Tools für die Studienorganisation."
    ),
    instrument=True,
)

# -------------------
# 6) Bibliothek - Agent
# -------------------
library_agent = Agent(
    model,
    deps_type=str,
    system_prompt=(
        "Du bist Experte für alle bibliotheksbezogenen Informationen: Öffnungszeiten, Kontakte, Kataloge (OPAC), "
        "digitale Zugänge, Datenbanken, Leihsysteme, aktuelle News sowie spezifische Hinweise für interne und "
        "externe Nutzer:innen."
    ),
    instrument=True,
)

# -------------------
# 7) Wissenschaftliches Arbeiten & Zitieren - Agent
# -------------------
academic_work_citation_agent = Agent(
    model,
    deps_type=str,
    system_prompt=(
        "Du berätst Studierende zu Software-Tools und Methoden rund um wissenschaftliches Arbeiten – insbesondere "
        "zum Zitieren mit Citavi oder Alternativen."
    ),
    instrument=True,
)

# -------------------
# 8) Rechenzentrum / IT-Support - Agent
# -------------------
it_support_agent = Agent(
    model,
    deps_type=str,
    system_prompt=(
        "Du beantwortest Fragen zum IT-Support für Studierende, wie aktuelle Systemmeldungen, FAQ, Remote-Zugriff "
        "auf Labor-PCs sowie andere rechenzentrumsnahe IT-Dienste."
    ),
    instrument=True,
)

# -------------------
# 9) Unterstützung & Services - Agent
# -------------------
support_services_agent = Agent(
    model,
    deps_type=str,
    system_prompt=(
        "Du kümmerst dich um übergeordnete Unterstützungsangebote wie Kompetenzzentren, spezialisierte Einrichtungen, "
        "Lehrunterstützung, Videoerstellung sowie Informationen vom Schwarzen Brett."
    ),
    instrument=True,
)


# -------------------
# Tools
# -------------------

# 1)
@search_research_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def websearch(ctx: RunContext, search_query: str, num_of_max_articles: int = 2) -> List[str]:
    """
    Searches the web using DuckDuckGo and extracts relevant information.

    This function queries DuckDuckGo using advanced search operators to refine results.
    The search can be fine-tuned with filters such as exact phrases, exclusions, file types,
    and domain-specific searches.

    Args:
        ctx (RunContext):
            The execution context for the tool.
        search_query (str):
            The search query, which can include DuckDuckGo search operators for more precise results.
        num_of_max_articles (int, default=2):
            The maximum number of articles to retrieve.

    Returns:
        List[str]: The cleaned and extracted text from the search results.
    """
    track_tool()
    logger.info(f"Using websearch with following search query: {search_query}")

    # Asynchrone Websearch durchführen
    search_results = full_duckduckgo_search(query=search_query, num_results=num_of_max_articles)
    logger.info("Extracting html..")
    extracted_html = await extract_html(search_results)
    logger.info("Cleaning html...")
    cleaned_html = await clean_html(extracted_html)

    return cleaned_html.content


# Tool 2: FAQ-Suche - Asynchron umgesetzt
@search_research_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_top_n_similar_questions_from_FAQ(ctx: RunContext, query: str, top_n: int = 5) -> List[
    Tuple[str, str, float]]:
    """
    Fetches the FAQ page from the OTH-AW site. Extract the Questions and calculates the cosine similarity to find the top 5 best questions.
    Source from the FAQ: https://www.oth-aw.de/studium/studienangebote/faq/

    Args:
        ctx (RunContext): The execution context for the tool.
        query (str): The question from the user.
        top_n (int): The number of similar questions to return. Defaults to 5.

    Returns:
        List[Tuple[str, str, float]]: A list of the top `top_n` similar questions with their answers and similarity scores.
    """
    track_tool()
    logger.info(f"Fetching FAQ page with following query: {query}")
    faq_data = await crawl_FAQ(query)
    logger.info("Looking for similar questions...")
    top_n_similar_questions = await find_similar_sentence(faq_data, query, top_n)

    return top_n_similar_questions


@search_research_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_specific_website(ctx: RunContext, url: str) -> str:
    """
    This tool extracts the main content from a OTH-AW webpage. Use this tool if the user gives you an url starting with: "https://www.oth-aw.de/"

    Args:
        ctx (RunContext): The runtime context provided during tool execution.
        url (str): The URL of the webpage from which the main content is to be extracted.

    Returns:
        str: Returns the cleaned text from the main content of the webpage, or an error message if the content cannot be found.
    """
    track_tool()
    logger.info(f"Calling a specific webpage: {url}")

    if "https://www.oth-aw.de/" in url:
        try:
            soup = await call_browser(url=url)
            try:
                side_section = soup.find("main")
                clean_webpage = side_section.get_text(strip=True)
            except AttributeError:
                clean_webpage = soup.get_text(strip=True)

            return clean_webpage
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return "An error happened please look at the main page: https://www.oth-aw.de/ "
    else:
        return "Error: This tool is only for crawling the https://www.oth-aw.de/ website."


# 2)

@study_programs_studies_in_general_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_study_program_overview(ctx: RunContext) -> List[Tuple[str, str]] | Exception:
    """
    Performs a web crawl to retrieve all study programs from the OTH-AW. For people who don't know what to study,
    there is a so-called “prepareING” orientation course where students can find out what they like in one semester.

    Args:
        ctx (RunContext): The execution context for the tool, containing relevant runtime information.

    Returns:
        List[Tuple[str, str]] | Exception: A list of tuples, each containing the name of a study program and its corresponding link.
        If unsuccessful, it returns an exception.
    """
    track_tool()
    logger.info("getting an overview of all study programs ...")
    ass = await all_study_programs()
    return ass


@study_programs_studies_in_general_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_specific_study_program(ctx: RunContext, specific_course_name: str) -> str:
    """
    Crawls more information about a specific study program from the OTH-AW.

    Available study programs include (but are not limited to):
        - Angewandte Wirtschaftspsychologie
        - Applied Research in Engineering Sciences
        - Arbeitsrecht (berufsbegleitend)
        - Artificial Intelligence for Industrial ApplicationsNeu
        - Betriebswirtschaft
        - Finance, Accounting, Controlling, Taxation (FACT)
        - Logistik & Supply Chain Management
        - Marketing & Vertrieb
        - Bio- und Umweltverfahrenstechnik
        - Digital Business
        - Digital Business
        - E-Commerce & Retail Management
        - Innovation & Process Management
        - Analytics & Data Management
        - Digital Business Management (berufsbegleitend)
        - Digital Entrepreneurship
        - Digital Healthcare Management
        - Digital Marketing (berufsbegleitend)
        - Digital Technology and Management
        - Educational Technology
        - Elektro- und Informationstechnik
        - Energietechnik, Energieeffizienz und Klimaschutz
        - Geoinformatik und Landmanagement
        - Global Research in Sustainable Engineering
        - Handels- und Dienstleistungsmanagement (berufsbegleitend)
        - Handels- und Gesellschaftsrecht (berufsbegleitend)
        - Industrial EngineeringNeu
        - Industrie-4.0.-Informatik
        - Ingenieurpädagogik Fachrichtung Elektro- und Informationstechnik
        - Ingenieurpädagogik Fachrichtung Metalltechnik
        - Innovationsfokussierter Maschinenbau
        - Interkulturelles Unternehmens- und Technologiemanagement
        - International Business
        - International Energy Engineering
        - International Management & Sustainability
        - Internationales Technologiemanagement
        - Product Life Cycle Management
        - Global Procurement & Sales
        - Digital Production & Logistics
        - International Management & Languages
        - IT und Automation
        - Künstliche Intelligenz
        - Künstliche Intelligenz
        - Künstliche Intelligenz – International
        - Logistik & Digitalisierung
        - Logistik & Digitalisierung
        - Maschinenbau
        - Mechatronik und digitale Automation
        - Medical EngineeringNeu
        - Medieninformatik
        - Medienproduktion und Medientechnik
        - Medientechnik und Medienproduktion
        - Medizinrecht (berufsbegleitend)
        - Medizintechnik
        - Digitale Medizintechnik
        - Medizinische Physik
        - Service & Application
        - Medizinische Produktentwicklung und Regulatory Affairs
        - Medizintechnik
        - Miet- und WEG-Recht (berufsbegleitend)
        - Motorsport Engineering
        - Physician Assistance - Arztassistenz
        - Physician Assistance – Arztassistenz für Gesundheitsfachberufe
        - prepareING
        - Steuerrecht und Steuerlehre (berufsbegleitend)
        - Technical EngineeringNeu
        - Technologiemanagement 4.0 (berufsbegleitend)
        - Umwelttechnologie
        - Wirtschaft und Recht
        - Wirtschaftsingenieurwesen
        - Digitale Produktentwicklung
        - Digitalisierung in Produktion und Logistik
        - Mobilität und Nachhaltigkeit
        - Wirtschaftsingenieurwesen - Digital Engineering & Management

    Args:
        ctx (RunContext): The execution context for the tool, containing relevant runtime information.
        specific_course_name (str): The name of the study program for which details should be retrieved.

    Returns:
        str: Information about the specific study program.
    """
    track_tool()
    logger.info("getting an overview of a specific study program: " + str(specific_course_name))
    aio = await get_more_details_about_specific_study_program(study_program=specific_course_name)
    logger.debug(f"Return from get_specific_study_program: {aio}")
    return aio


@study_programs_studies_in_general_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_overview_of_one_faculty(ctx: RunContext, which_faculty: str) -> List[str]:
    """
    There are four faculties. To get an overview of a faculty, enter one of the following four values: "Electrical_Engineering_Media_Informatics", "Mechanical_Engineering_Environmental_Engineering", "Weiden_Business_School" or "Industrial_Engineering_Health".

    Args:
        ctx (): The execution context for the tool.
        which_faculty (): four options: "Electrical_Engineering_Media_Informatics", "Mechanical_Engineering_Environmental_Engineering", "Weiden_Business_School" or "Industrial_Engineering_Health"

    Returns: Returns an overview over the given faculty as a list containing strings.

    """
    track_tool()
    logger.info("Using get_overview_of_one_faculty...")
    if which_faculty == "Electrical_Engineering_Media_Informatics":
        logger.info("Electrical Engineering Media...")
        return await crawl_elektrotechnik_medien_informatik_overview()
    elif which_faculty == "Mechanical_Engineering_Environmental_Engineering":
        logger.info("Mechanical Engineering...")
        return await crawl_maschinenbau_umwelttechnik_overview()
    elif which_faculty == "Weiden_Business_School":
        logger.info("Weiden Business School...")
        return await crawl_weiden_business_school_overview()
    elif which_faculty == "Industrial_Engineering_Health":
        logger.info("Industrial..")
        return await crawl_wirtschaft_gesundheit_overview()
    else:
        return ['Only "Electrical_Engineering_Media_Informatics", "Mechanical_Engineering_Environmental_Engineering", '
                '"Weiden_Business_School" or "Industrial_Engineering_Health" are allowed as parameter.']


@study_programs_studies_in_general_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_specific_study_schedule_stundenplan(ctx: RunContext, which_study_program: str) -> list[tuple[str, str, float]]:
    """
    Get a specific study schedule (in german: stundenplan) from the OTH-AW. returns the two most likely schedules. This are the available study programms: Bachelor Elektro- und Informationstechnik, Bachelor Geoinformatik und Landmanagement, Bachelor Industrie-4.0-Informatik, Ingenieurpädagogik – Fachrichtung Elektro- und Informationstechnik, Bachelor Künstliche Intelligenz, Bachelor Künstliche Intelligenz – International, Bachelor Medieninformatik, Bachelor Medienproduktion und Medientechnik, Master Educational Technology, Master IT und Automation, Master Artificial Intelligence for Industrial Applications, Master Künstliche Intelligenz, Master Medientechnik und Medienproduktion, Bio- und Umweltverfahrenstechnik, Energietechnik, Energieeffizienz und Klimaschutz, International Energy Engineering, Ingenieurpädagogik – berufliche Fachrichtung Metalltechnik, Innovationsfokussierter Maschinenbau, Kunststofftechnik, Maschinenbau, Mechatronik und digitale Automation, Motorsport Engineering, Patentingenieurwesen, Technical Engineering, Umwelttechnologie, Bachelor Angewandte Wirtschaftspsychologie, Bachelorstudiengang Betriebswirtschaft, Bachelor Digital Business, Handels- und Dienstleistungsmanagement berufsbegleitend, Bachelorstudiengang Handels- und Dienstleistungsmanagement, Bachelorstudiengang International Business, Bachelorstudiengang Logistik und Digitalisierung, Masterstudiengang Angewandte Wirtschaftspsychologie, Masterstudiengang Digital Business, Master Digital Entrepeneurship, Masterstudiengang Logistik & Digitalisierung, Master International Management & Sustainability, Bachelor Digital Healthcare Management, Bachelor Digital Technology and Management, Bachelor Industrial Engineering, Bachelor Medical Engineering, Bachelor Medizintechnik, Bachelor Physician Assistance - Arztassistenz, Bachelor Physician Assistance - Arztassistenz für Gesundheitsfachberufe, Bachelor Internationales Technologiemanagement, Bachelor Wirtschaftsingenieurwesen, Master Medizintechnik, Master Interkulturelles Unternehmens- und Technologiemanagement, Master Wirtschaftsingenieurwesen - Digital Engineering & Management

    Args:
        ctx (): The execution context for the tool.
        which_study_program (): search for a specific study schedule.

    Returns: the most two likely study schedules as string

    """
    track_tool()
    logger.info("crawling " + which_study_program + "...")
    a = await crawl_stundenplan()
    logger.info("finding the most similar study programms...")
    return await find_similar_sentence(data=a, query=which_study_program)


@study_programs_studies_in_general_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_specific_study_examination_plans_pruefungsplan(ctx: RunContext, which_study_program: str) -> list[
    tuple[str, str, float]]:
    """
    Get a specific Examination plans (in german: Prüfungsplan) from the OTH-AW. returns the two most likely exam plans. This are the available study programms: Bachelor Elektro- und Informationstechnik, Bachelor Geoinformatik und Landmanagement, Bachelor Industrie-4.0-Informatik, Ingenieurpädagogik – Fachrichtung Elektro- und Informationstechnik, Bachelor Künstliche Intelligenz, Bachelor Künstliche Intelligenz – International, Bachelor Medieninformatik, Bachelor Medienproduktion und Medientechnik, Master Educational Technology, Master IT und Automation, Master Artificial Intelligence for Industrial Applications, Master Künstliche Intelligenz, Master Medientechnik und Medienproduktion, Bio- und Umweltverfahrenstechnik, Energietechnik, Energieeffizienz und Klimaschutz, International Energy Engineering, Ingenieurpädagogik – berufliche Fachrichtung Metalltechnik, Innovationsfokussierter Maschinenbau, Kunststofftechnik, Maschinenbau, Mechatronik und digitale Automation, Motorsport Engineering, Patentingenieurwesen, Technical Engineering, Umwelttechnologie, Bachelor Angewandte Wirtschaftspsychologie, Bachelorstudiengang Betriebswirtschaft, Bachelor Digital Business, Handels- und Dienstleistungsmanagement berufsbegleitend, Bachelorstudiengang Handels- und Dienstleistungsmanagement, Bachelorstudiengang International Business, Bachelorstudiengang Logistik und Digitalisierung, Masterstudiengang Angewandte Wirtschaftspsychologie, Masterstudiengang Digital Business, Master Digital Entrepeneurship, Masterstudiengang Logistik & Digitalisierung, Master International Management & Sustainability, Bachelor Digital Healthcare Management, Bachelor Digital Technology and Management, Bachelor Industrial Engineering, Bachelor Medical Engineering, Bachelor Medizintechnik, Bachelor Physician Assistance - Arztassistenz, Bachelor Physician Assistance - Arztassistenz für Gesundheitsfachberufe, Bachelor Internationales Technologiemanagement, Bachelor Wirtschaftsingenieurwesen, Master Medizintechnik, Master Interkulturelles Unternehmens- und Technologiemanagement, Master Wirtschaftsingenieurwesen - Digital Engineering & Management

    Args:
        ctx (): The execution context for the tool.
        which_study_program (): search for a specific examination plan.

    Returns: the most two likely examination plans as string

    """
    track_tool()
    logger.info("crawling " + which_study_program + "...")
    a = await crawl_pruefungsplaene()
    logger.info("finding the most similar examination plans...")
    return await find_similar_sentence(data=a, query=which_study_program)


@study_programs_studies_in_general_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_information_change_study_programm_study_other_topic(ctx: RunContext) -> List[str]:
    """
    Provides information for individuals interested in changing their degree program.

    Args:
        ctx (RunContext): The execution context for the tool, containing relevant runtime information.

    Returns:
        List[str]: Information regarding the process of changing the degree program.
    """
    track_tool()
    logger.info("crawling studienablauf -> studiengangswechsel ...")
    a = await studiengangswechsel()
    return a


@study_programs_studies_in_general_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_information_on_taking_a_leave_of_absence_during_study(ctx: RunContext) -> List[str]:
    """
    Provides information on taking a leave of absence during study.

    Args:
        ctx (RunContext): The execution context for the tool, containing relevant runtime information.

    Returns:
        List[str]: Information regarding the leave of absence during studies.
    """
    track_tool()
    logger.info("crawling studienablauf -> beurlaubung ...")
    a = await beurlaubung()
    return a


@study_programs_studies_in_general_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_information_exmatriculation(ctx: RunContext) -> List[str]:
    """
    Here you will find information about the de-registration process.

    Args:
        ctx (RunContext): The execution context for the tool, containing relevant runtime information.

    Returns:
        List[str]: Information about the de-registration process.
    """
    track_tool()
    logger.info("crawling studienablauf -> exmatrikulation ...")
    a = await exmatrikulation()
    return a


@study_programs_studies_in_general_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_information_for_students_re_registration(ctx: RunContext) -> List[str]:
    """
    Here you will find information about the re-registration (Rückmeldung zum nächsten Semester) process and its associated costs.

    Args:
        ctx (RunContext): The execution context for the tool, containing relevant runtime information.

    Returns:
        List[str]: Information about the re-registration process and the associated costs.
    """
    track_tool()
    logger.info("crawling studienablauf -> rueckmeldungen_studentenwerksbeitrag ...")
    a = await rueckmeldungen_studentenwerksbeitrag()
    return a


# 3)

@forms_documents_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_internship_semester_praxissemester_documents(ctx: RunContext) -> List[str]:
    """
    Get the documents for the internship semester/practical semester and returns the contact data of the responsible study office.

    Args:
        ctx (RunContext): The execution context for the tool, containing relevant runtime information.

    Returns:
        List[str]: The links for the documents and contact data of the responsible study office.
    """
    track_tool()
    logger.info("crawling studienablauf -> praxissemester ...")
    a = await praxissemester()
    return a


@forms_documents_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_forms_guidelines_bachelor_master_thesis_documents(ctx: RunContext) -> List[str]:
    """
    Gets forms and guidelines for writing the Bachelor's or Master's thesis at the OTH Amberg-Weiden.

    Args:
        ctx (RunContext): The execution context for the tool, containing relevant runtime information.

    Returns:
        List[str]: A summary and the necessary documents for writing the Bachelor's or Master's thesis at the OTH-AW.
    """
    track_tool()
    logger.info("crawling studienablauf -> abschlussarbeiten ...")
    a = await abschlussarbeiten()
    return a


# 4)

@personnel_contacts_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_employee_professor_info(ctx: RunContext, searchterm: str) -> List[dict] | Exception:
    """
    This tool can search specifically for employees, responsible persons or teaching areas. The tool works by
    requiring a search term to be entered. This can either be the name of an employee or professor, or a subject area
    such as electrical engineering. Depending on what you are looking for, you will get back the relevant contact
    details. It is also possible to search for administrative staff by searching for “Studienbüro”, for example.
    Here, for example, you would get back all the contact persons who deal with administrative matters. If the person
    is looking for specific persons related to exams so the "Studienbüro" should help. The university has two
    locations. It is always best if the links to the profile URLs are included in the answer so that the user can
    check again.

    Args:
        ctx (RunContext): The runtime context provided during tool execution.
        searchterm (str): The term to search for (e.g., a name or department keyword).

    Returns:
        List[dict] | Exception: Returns a list of dictionaries containing contact data information,
        or an Exception if something goes wrong.
    """
    track_tool()
    logger.info(f"searching employees with this search term... {searchterm}")
    a = get_team(search_term=searchterm)
    return a


@personnel_contacts_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_details_about_prof(ctx: RunContext, prof_name: str) -> dict[str, str] | dict | tuple[
    list[tuple[str, str]], str] | list[tuple[str, str]]:
    """
    This tool extracts general information about a professor. The tool returns a short snipped about the information
    but also a list of links to topics related to the prof (because the information are all on the profs page).

    Args:
        ctx (RunContext): The runtime context provided during tool execution.
        prof_name (str): The name of the professor to search for.

    Returns: list[tuple[str, str]] | str: Returns a list of tuples containing extracted URLs and titles from the
    professor's page, or a string message.
    """
    track_tool()
    logger.info(f"searching specific prof: {prof_name}")
    prof_page = get_team(search_term=prof_name)

    if not prof_page:
        logger.error("get_team hat keine Ergebnisse zurückgegeben (leere Liste).")
        # Hier solltest du eine passende Fehlermeldung zurückgeben
        return {"Error": "Für diesen Suchbegriff wurden keine Personen gefunden."}

    first_result = prof_page[0]

    if "Error" in first_result:
        logger.warning(f"get_team hat einen Fehler zurückgegeben: {first_result['Error']}")
        return first_result

    try:
        profile_url = first_result["Profil-URL"]  # Zugriff ist jetzt sicher

        logger.info(f"Rufe Browser für URL auf: {profile_url}")
        webpage = await call_browser(url=profile_url)

        side_section = webpage.find("main")
        clean_webpage = side_section.get_text(strip=True)
        return await extract_links_with_titles(prof_page[0]["Profil-URL"]), clean_webpage
    except:
        return await extract_links_with_titles(prof_page[0]["Profil-URL"])


# 5)

@campus_services_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def kursanmeldung_link(ctx: RunContext) -> str:
    """
    Gibt einen Hinweistext zur Anmeldung von Wahlpflichtfächern mit entsprechendem Link zurück.

    Bei einigen Studiengängen besteht die Möglichkeit, Wahlpflichtfächer zu belegen. Sobald diese zur Anmeldung freigegeben
    sind, kann die Kursanmeldung über die verlinkte Seite erfolgen. Da einige Wahlpflichtfächer sehr beliebt sind, wird empfohlen,
    sich möglichst frühzeitig anzumelden.

    Args:
        ctx (RunContext): Der aktuelle Ausführungskontext des Agenten. Wird für diese Funktion nicht verwendet, ist aber erforderlich für die Tool-Signatur.

    Returns:
        str: Ein informativer Hinweistext zur Kursanmeldung mit dem entsprechenden Link.
    """
    info = (
        "Bei manchen Studiengängen kann man Wahlpflichtfächer absolvieren. Sobald diese freigeschaltet sind, "
        "kann man sich über diesen Link anmelden: "
        "https://www.oth-aw.de/hochschule/services/online-services/kursanmeldung/\n\n"
        "Tipp: Einige Wahlpflichtfächer sind sehr beliebt – daher wird empfohlen, sich frühzeitig anzumelden. 😉"
    )
    return info


@campus_services_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def primuss_campus_link(ctx: RunContext) -> str:
    """
    Gibt eine Übersicht über die Funktionen des PRIMUSS-Portals inklusive Login-Link.

    Über das PRIMUSS-Portal der OTH Amberg-Weiden kannst du viele studienrelevante Informationen einsehen und Services nutzen.
    Dazu gehören unter anderem die Rückmeldung zum Semester, das Herunterladen von Bescheinigungen sowie die Einsicht von Noten
    und Prüfungsanmeldungen.

    Args:
        ctx (RunContext): Der aktuelle Ausführungskontext des Agenten. Wird für diese Funktion nicht aktiv genutzt,
                          ist aber erforderlich für die Tool-Signatur.

    Returns:
        str: Ein informativer Text mit dem Link zum PRIMUSS-Portal und einer Beschreibung der wichtigsten Funktionen.
    """
    info = (
        "Wähle im Portal einfach die OTH Amberg-Weiden aus, dann wirst du zur Anmeldemaske weitergeleitet 🤗\n"
        "Hier ist der Link: https://www3.primuss.de/cgi-bin/login/index.pl\n\n"
        "Im PRIMUSS-Portal findest du:\n"
        "- **Status**: Sieh nach, ob du aktuell eingeschrieben bist.\n"
        "- **Studienbüro**: Rückmeldung für das kommende Semester und Bescheinigungen (z.B. Immatrikulationsbescheinigung).\n"
        "- **Prüfungsamt**: Aktuelles Notenblatt, angemeldete Prüfungen (manuelle Anmeldung erforderlich), und deine Noten "
        "für das laufende Semester.\n"
        "- **Pflichtpraktikum**: Anmeldung, falls in deinem Studiengang vorgesehen.\n"
        "- Weitere Bereiche: Mutterschutz, Benutzerkonto, Anträge usw.\n\n"
        "Das Portal ist dein zentraler Zugangspunkt für viele organisatorische Themen rund ums Studium."
    )
    return info


@campus_services_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_wifi_eduroam_vpn_info(ctx: RunContext) -> str:
    """
    Retrieves information about WiFi (eduroam), VPN, and WLAN at the OTH.

    This tool provides students and staff with information on how to access
    the internet via eduroam, configure VPN access, and connect to wireless networks
    at the OTH Amberg-Weiden.

    Args:
        ctx (RunContext): The current runtime context of the LLM call.

    Returns:
        str: A string containing the network access and VPN instructions.
    """
    logger.info("Fetching eduroam, WLAN, and VPN info...")
    try:
        result = await get_wifi_eduroam_vpn_info_live()
        logger.debug(f"Retrieved network info: {result[:100]}...")  # Truncate to avoid huge logs
        return result
    except Exception as e:
        logger.error("Failed to retrieve eduroam/VPN info", exc_info=True)
        raise


@campus_services_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_email_contact_calender(ctx: RunContext) -> str:
    """
    Retrieves information about email, contact, and calendar systems at the OTH.

    This tool explains how to access Webmail, manage contacts, use the calendar,
    and synchronize with external devices or clients.

    Args:
        ctx (RunContext): The current runtime context of the LLM call.

    Returns:
        str: A description of the email, contacts, and calendar system.
    """
    logger.info("Fetching email/contact/calendar info...")
    try:
        result = await get_email_contact_calender_live()
        logger.debug(f"Retrieved info: {result[:100]}...")
        return result
    except Exception as e:
        logger.error("Failed to retrieve email/contact/calendar info", exc_info=True)
        raise


@campus_services_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_info_for_file_exchange(ctx: RunContext) -> str:
    """
    Retrieves information on file exchange options within the OTH-AW network.

    This includes instructions on how to securely share files with others
    inside the university infrastructure.

    Args:
        ctx (RunContext): The current runtime context of the LLM call.

    Returns:
        str: File exchange instructions and tools available at the OTH.
    """
    logger.info("Fetching file exchange info...")
    try:
        result = await get_file_exchange()
        logger.debug(f"Retrieved file exchange info: {result[:100]}...")
        return result
    except Exception as e:
        logger.error("Failed to retrieve file exchange info", exc_info=True)
        raise


@campus_services_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_info_for_printing_on_the_oth(ctx: RunContext) -> str:
    """
    Retrieves information on printing services at the OTH.

    This includes printer locations, configuration, and how the print system works
    across different departments and campuses.

    Args:
        ctx (RunContext): The current runtime context of the LLM call.

    Returns:
        str: Printing-related instructions and system details.
    """
    logger.info("Fetching printing info...")
    try:
        result = await get_print_info()
        logger.debug(f"Retrieved printing info: {result[:100]}...")
        return result
    except Exception as e:
        logger.error("Failed to retrieve printing info", exc_info=True)
        raise


@campus_services_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_site_map_of_different_location_amberg_weiden(ctx: RunContext) -> List[str]:
    """
    Retrieves site maps for the Amberg and Weiden campuses, including room and office locations.

    Useful for new students or visitors trying to locate lecture halls, professor offices,
    and administrative departments.

    Args:
        ctx (RunContext): The current runtime context of the LLM call.

    Returns:
        List[str]: A list of links or descriptions related to campus maps and room finder tools.
    """
    logger.info("Fetching campus site maps for Amberg/Weiden...")
    try:
        result = lageplan_raumfinder()
        logger.debug(f"Retrieved site map info: {result}")
        return result
    except Exception as e:
        logger.error("Failed to retrieve site map info", exc_info=True)
        raise


@campus_services_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_Vorlesungs_Pruefungs_und_vorlesungsfreie_Zeiten(ctx: RunContext) -> dict[str, dict[str, str]] | str:
    """
    Gibt folgende wichtige Termine zurück: Semesterbeginn, Vorlesungszeit, Erstsemesterbegrüßung, Rückmeldung für nachfolgendes Semester, Prüfungszeit, Anmeldung zu den Prüfungen, Notenbekanntgabe, Vorlesungsfreie Zeiten.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        Dict[str, Dict[str, str]] | str: returns a dict with all necessary information's.
    """
    try:
        return await get_wichtige_termine()
    except:
        return "An error occoured. Use the websearch with following url: https://www.oth-aw.de/studium/im-studium/organisatorisches/vorlesungs-pruefungs-vorlesungsfreie-zeiten/"


@campus_services_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_OTH_AW_app_information(ctx: RunContext) -> dict | str:
    """
    Gibt Informationen über die OTH-AW App zurück.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        dict | str: returns a dict with all necessary information's.
    """

    return await oth_aw_app()


@campus_services_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_terminplaner_link(ctx: RunContext) -> str:
    """
    Gibt den link des offiziellen Terminplaners zurück.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        str: returns a str with all necessary information's.
    """

    return await oth_aw_terminplaner()


@campus_services_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_gitlab_link(ctx: RunContext) -> str:
    """
    Returns the url to the Gitlab (GitHub replacement) of the OTH-AW university.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        str: url of the gitlab.
    """
    return "https://git.oth-aw.de/users/sign_in"


# 6)

@library_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_contact_data_of_library(ctx: RunContext) -> str:
    """
    Retrieves contact information for the OTH library.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        str: Contact details including phone, email, and address.
    """
    logger.info("Fetching library contact data...")
    try:
        result = await bibliothek_kontakt()
        logger.debug(f"Library contact: {result}")
        return result
    except Exception as e:
        logger.error("Failed to get library contact data", exc_info=True)
        raise


@library_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_current_news_from_library_relevant_for_students(ctx: RunContext) -> List[str] | str:
    """
    Retrieves current news from the library relevant for students.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        List[str] | str: A list or string of news entries.
    """
    logger.info("Fetching general library news for students...")
    try:
        result = await bibliothek_aktuelles()
        logger.debug(f"Library news: {result}")
        return result
    except Exception as e:
        logger.error("Failed to retrieve student-relevant library news", exc_info=True)
        raise


@library_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_current_news_from_library_specific(ctx: RunContext, headers: str) -> str:
    """
    Retrieves a specific news article from the library by title.

    Note:
        Must be called after `get_current_news_from_library`,
        as valid headers must come from its output.

    Args:
        ctx (RunContext): The current runtime context.
        headers (str): The header/title of the specific news entry.

    Returns:
        str: The detailed news content.
    """
    logger.info(f"Fetching specific library news for header: {headers}")
    try:
        result = await bibliothek_aktuelles_specific(headers)
        logger.debug(f"Specific news content: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to get specific news with header '{headers}'", exc_info=True)
        raise


@library_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_current_news_events_from_library_more_informations(ctx: RunContext) -> List[str] | str:
    """
    Retrieves further details about current events/news from the library.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        List[str] | str: List or string of news/events with more information.
    """
    logger.info("Fetching detailed library news/events...")
    try:
        result = await bibliothek_news()
        logger.debug(f"Detailed news: {result}")
        return result
    except Exception as e:
        logger.error("Failed to retrieve more detailed library news", exc_info=True)
        raise


@library_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_library_opening_hours(ctx: RunContext) -> str:
    """
    Retrieves the regular opening hours of the library.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        str: The opening times of the library.
    """
    logger.info("Fetching library opening hours...")
    try:
        result = await bibliothek_oeffnungszeiten()
        logger.debug(f"Library hours: {result}")
        return result
    except Exception as e:
        logger.error("Failed to get library opening hours", exc_info=True)
        raise


@library_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_library_24h_opening_info(ctx: RunContext) -> str:
    """
    Provides information about 24-hour library access (if available).

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        str: Information about 24h access policies.
    """
    logger.info("Fetching 24h access info for the library...")
    try:
        result = await bibliothek_24h_open()
        logger.debug(f"24h info: {result}")
        return result
    except Exception as e:
        logger.error("Failed to retrieve 24h library access info", exc_info=True)
        raise


@library_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_library_team_contact_data(ctx: RunContext) -> list[dict[str, str | LiteralString | Any]] | str:
    """
    Retrieves contact information of the library team.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        list[dict[str, str | LiteralString | Any]] | str: Contact list or error message.
    """
    logger.info("Fetching library team contact info...")
    try:
        result = await bibliothek_team()
        logger.debug(f"Library team contacts: {result}")
        return result
    except Exception as e:
        logger.error("Failed to retrieve library team contact data", exc_info=True)
        raise


@library_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_library_note_as_external_user(ctx: RunContext) -> str:
    """
    Provides notes or requirements for external users of the library.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        str: Information for external users.
    """
    logger.info("Fetching library note for external users...")
    try:
        result = await bibliothek_hinweis_als_externer_nutzer()
        logger.debug(f"External user note: {result}")
        return result
    except Exception as e:
        logger.error("Failed to get external user info", exc_info=True)
        raise


@library_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_library_OPAC_Online_Public_Access_Catalogue_info(ctx: RunContext) -> str:
    """
    Provides information about the OPAC system of the library.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        str: OPAC usage guide and access info.
    """
    logger.info("Fetching OPAC info...")
    try:
        result = await bibliothek_OPAC_Online_Public_Access_Catalogue()
        logger.debug(f"OPAC info: {result}")
        return result
    except Exception as e:
        logger.error("Failed to get OPAC info", exc_info=True)
        raise


@library_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_digital_library_info(ctx: RunContext) -> str:
    """
    Provides information about the digital library services.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        str: Digital library resources and access methods.
    """
    logger.info("Fetching digital library info...")
    try:
        result = await bibliothek_digitale_bibliothek()
        logger.debug(f"Digital library info: {result}")
        return result
    except Exception as e:
        logger.error("Failed to retrieve digital library info", exc_info=True)
        raise


@library_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_library_list_of_databases(ctx: RunContext) -> str:
    """
    Retrieves a list of databases available via the library.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        str: List of available databases.
    """
    logger.info("Fetching list of library databases...")
    try:
        result = await bibliothek_datenbanken()
        logger.debug(f"Library databases: {result}")
        return result
    except Exception as e:
        logger.error("Failed to retrieve list of databases", exc_info=True)
        raise


@library_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_library_access_to_journals_info(ctx: RunContext) -> str:
    """
    Provides access information to online and physical journals.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        str: Journal access instructions.
    """
    logger.info("Fetching journal access info...")
    try:
        result = await bibliothek_zeitschriften()
        logger.debug(f"Journal access: {result}")
        return result
    except Exception as e:
        logger.error("Failed to retrieve journal access info", exc_info=True)
        raise


@library_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_library_lend_info(ctx: RunContext) -> str:
    """
    Provides information about borrowing books and media.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        str: Lending rules and instructions.
    """
    logger.info("Fetching lending info...")
    try:
        result = await bibliothek_ausleihen()
        logger.debug(f"Lending info: {result}")
        return result
    except Exception as e:
        logger.error("Failed to retrieve lending info", exc_info=True)
        raise


@library_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_general_question_for_library(ctx: RunContext, user_question: str) -> list[tuple[str, str, float]]:
    """
    Handles general user questions regarding the library using a FAQ system.

    Args:
        ctx (RunContext): The current runtime context.
        user_question (str): The question posed by the user.

    Returns:
        list[tuple[str, str, float]]: Matching FAQs and their relevance scores.
    """
    logger.info(f"Processing library FAQ for question: {user_question}")
    try:
        result = await bibliothek_FAQ(user_question)
        logger.debug(f"FAQ result: {result}")
        return result
    except Exception as e:
        logger.error("Failed to process FAQ question", exc_info=True)
        raise


# 7)

@academic_work_citation_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_citation_tool_info_citavi(ctx: RunContext) -> str:
    """
    Provides information about using Citavi as a citation tool.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        str: Instructions and support for Citavi.
    """
    logger.info("Fetching Citavi citation tool info...")
    try:
        result = await bibliothek_zitiertool_citavi()
        logger.debug(f"Citavi info: {result}")
        return result
    except Exception as e:
        logger.error("Failed to retrieve Citavi info", exc_info=True)
        raise


@academic_work_citation_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_citation_tool_info_alternatives(ctx: RunContext) -> str:
    """
    Provides information about alternative citation tools to Citavi.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        str: Overview of alternative tools.
    """
    logger.info("Fetching alternative citation tool info...")
    try:
        result = await bibliothek_zitiertool_alternativen()
        logger.debug(f"Alternative citation tools: {result}")
        return result
    except Exception as e:
        logger.error("Failed to retrieve alternative citation tools", exc_info=True)
        raise


# 8 )

@it_support_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_rechenzentrum_data_center_news_for_students(ctx: RunContext) -> str:
    """
    Retrieves current IT service news from the data center relevant to students.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        str: News from the data center.
    """
    logger.info("Fetching data center news for students...")
    try:
        result = await rechenzentrum_news()
        logger.debug(f"Data center news: {result}")
        return result
    except Exception as e:
        logger.error("Failed to retrieve data center news", exc_info=True)
        raise


@it_support_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_rechenzentrum_FAQ(ctx: RunContext, category: str) -> str:
    """
    Retrieves an FAQ entry from the data center (Rechenzentrum) for a specific category.

    Only the following categories are allowed (select exactly one):

    Only following categories/questions are allowed:
    Wähle das passendste aus dieser Liste aus für category:
    Wie bekomme ich einen Benutzeraccount?
    Warum kann ich mich nicht mehr anmelden?
    Warum hat sich meine Kennung geändert?
    Was passiert mit meiner Kennung nach der Exmatrikulation bzw. nach dem Ausscheiden aus dem Dienst?
    Wie ändere ich mein Passwort?
    Adobe Passwort zurücksetzen
    Wie setze ich mein Passwort zurück?
    Wie ergänze ich meinen Account mit einer privaten E-Mail Adresse und Mobilfunknummer?
    Eduroam über Easyroam
    WLAN Troubleshooting
    Wie konfiguriere ich den GroupWise-Client?
    Werden meine E-Mails gesichert?
    Welche Servereinstellungen sind für alternative E-Mail-Clients notwendig?
    Warum kann man keine Nachrichten von fremden Websites an die Hochschule senden lassen?
    Unerwünschte Mails (SPAM) in meinem Hochschulpostfach.
    Kann ich meine E-Mails auch ohne Mailprogramm lesen?
    Wie groß ist das E-Mail Postfach?
    Wie kann ich mein Postfach mit dem Smartphone abrufen?
    Wie groß darf eine E-Mail maximal sein?
    E-Mail Verteilerlisten für Wissenschaft und Forschung
    Wie leite ich E-Mails als Anlage weiter?
    Wie stelle ich das Format zum Verschicken und Empfangen von E-Mails um?
    Wie überprüfe ich den Absender einer E-Mail?
    Wie verschicke und empfange ich E-Mails im Text-Format?
    Was bedeuten die OpenText Rechte auf Dateien und Ordnern?
    Das L-Laufwerk ist das des anderen Standorts.
    Kann man vom Netzlaufwerk gelöschte Dateien wiederherstellen?
    Wo kann ich temporär größere Dateien ablegen?
    Wie groß ist die Kapazität des persönlichen Laufwerks?
    Myfiles Filr
    GigaMove 2.0
    Filr Client Anmeldung
    Wie nutze ich die Fax to Mail bzw. Mail To Fax Funktion?
    Wie ändere ich die PIN meines AVAYA Telefons?
    Wo findet man die Bedienungsanleitungen zu den Avaya Telefonen?
    Abwesenheitsregel einstellen
    Wie Faxe ich mit Groupwise?
    Wieso verschwinden meine Dateinen vom C: Laufwerk?
    Wie kann man eine Fehlermeldung zur weiteren Behebung besser dokumentieren?
    Nachrüsten des Microsoft Store
    Leeren des Browsercache für bestimmte Seiten
    Wie überprüfe ich einen Link/Button?
    Welche Dateiformate können potenziell gefährlich sein?
    Apple Mail einrichten
    Kalender und Kontakte einrichten
    Wie erhalte ich die direkten CalDAV/CarDAV Pfade?
    Bekannte Probleme
    Generelles
    Datenschutz
    Kostenrisiko
    Datenverlust
    Zuverlässigkeit des Dienstes
    Synchronisation bei Android einrichten
    Synchronisation bei iPhone / iPad einrichten
    Rocket.Chat
    Ende zu Ende Verschlüsselung im RocketChat
    Wie stelle ich eine Zertifikatsanfrage?
    Wie setze ich mein Office 365 Passwort zurück?
    Wie bekomme ich Zugang zu Office 365?
    Ihr Konto lässt keine Bearbeitung auf einem Mac zu
    Microsoft Azure Dev Tools for Teaching
    FortiClient VPN auf Windows
    FortiClient VPN auf Linux
    FortiClient VPN auf MacOS
    Welche Software ist auf den Rechnern in den Laboren in Amberg installiert?
    Welche Software ist auf den Rechnern in den Laboren in Weiden installiert?
    Fernzugriff / Splashtop

    Args:
        ctx (RunContext): The current runtime context.
        category (str): The question/category from the official FAQ list.

    Returns:
        str: The answer for the requested FAQ category.

    Raises:
        ValueError: If the category is not part of the predefined list.
    """

    ALLOWED_CATEGORIES = [
        "Wie bekomme ich einen Benutzeraccount?",
        "Warum kann ich mich nicht mehr anmelden?",
        "Warum hat sich meine Kennung geändert?",
        "Was passiert mit meiner Kennung nach der Exmatrikulation bzw. nach dem Ausscheiden aus dem Dienst?",
        "Wie ändere ich mein Passwort?",
        "Adobe Passwort zurücksetzen",
        "Wie setze ich mein Passwort zurück?",
        "Wie ergänze ich meinen Account mit einer privaten E-Mail Adresse und Mobilfunknummer?",
        "Eduroam über Easyroam",
        "WLAN Troubleshooting",
        "Wie konfiguriere ich den GroupWise-Client?",
        "Werden meine E-Mails gesichert?",
        "Welche Servereinstellungen sind für alternative E-Mail-Clients notwendig?",
        "Warum kann man keine Nachrichten von fremden Websites an die Hochschule senden lassen?",
        "Unerwünschte Mails (SPAM) in meinem Hochschulpostfach.",
        "Kann ich meine E-Mails auch ohne Mailprogramm lesen?",
        "Wie groß ist das E-Mail Postfach?",
        "Wie kann ich mein Postfach mit dem Smartphone abrufen?",
        "Wie groß darf eine E-Mail maximal sein?",
        "E-Mail Verteilerlisten für Wissenschaft und Forschung",
        "Wie leite ich E-Mails als Anlage weiter?",
        "Wie stelle ich das Format zum Verschicken und Empfangen von E-Mails um?",
        "Wie überprüfe ich den Absender einer E-Mail?",
        "Wie verschicke und empfange ich E-Mails im Text-Format?",
        "Was bedeuten die OpenText Rechte auf Dateien und Ordnern?",
        "Das L-Laufwerk ist das des anderen Standorts.",
        "Kann man vom Netzlaufwerk gelöschte Dateien wiederherstellen?",
        "Wo kann ich temporär größere Dateien ablegen?",
        "Wie groß ist die Kapazität des persönlichen Laufwerks?",
        "Myfiles Filr",
        "GigaMove 2.0",
        "Filr Client Anmeldung",
        "Wie nutze ich die Fax to Mail bzw. Mail To Fax Funktion?",
        "Wie ändere ich die PIN meines AVAYA Telefons?",
        "Wo findet man die Bedienungsanleitungen zu den Avaya Telefonen?",
        "Abwesenheitsregel einstellen",
        "Wie Faxe ich mit Groupwise?",
        "Wieso verschwinden meine Dateinen vom C: Laufwerk?",
        "Wie kann man eine Fehlermeldung zur weiteren Behebung besser dokumentieren?",
        "Nachrüsten des Microsoft Store",
        "Leeren des Browsercache für bestimmte Seiten",
        "Wie überprüfe ich einen Link/Button?",
        "Welche Dateiformate können potenziell gefährlich sein?",
        "Apple Mail einrichten",
        "Kalender und Kontakte einrichten",
        "Wie erhalte ich die direkten CalDAV/CarDAV Pfade?",
        "Bekannte Probleme",
        "Generelles",
        "Datenschutz",
        "Kostenrisiko",
        "Datenverlust",
        "Zuverlässigkeit des Dienstes",
        "Synchronisation bei Android einrichten",
        "Synchronisation bei iPhone / iPad einrichten",
        "Rocket.Chat",
        "Ende zu Ende Verschlüsselung im RocketChat",
        "Wie stelle ich eine Zertifikatsanfrage?",
        "Wie setze ich mein Office 365 Passwort zurück?",
        "Wie bekomme ich Zugang zu Office 365?",
        "Ihr Konto lässt keine Bearbeitung auf einem Mac zu",
        "Microsoft Azure Dev Tools for Teaching",
        "FortiClient VPN auf Windows",
        "FortiClient VPN auf Linux",
        "FortiClient VPN auf MacOS",
        "Welche Software ist auf den Rechnern in den Laboren in Amberg installiert?",
        "Welche Software ist auf den Rechnern in den Laboren in Weiden installiert?",
        "Fernzugriff / Splashtop"
    ]
    logger.info(f"Fetching FAQ for category: '{category}'")

    if category not in ALLOWED_CATEGORIES:
        logger.error(f"Invalid FAQ category: '{category}'", exc_info=True)
        raise ValueError(
            f"Ungültige Kategorie: '{category}'. "
            "Bitte wähle eine gültige Kategorie aus der FAQ-Liste aus."
        )

    try:
        result = await rechenzentrum_FAQ(category)
        logger.debug(f"FAQ result: {result}")
        return result
    except Exception as e:
        logger.error("Failed to fetch FAQ from data center", exc_info=True)
        raise


@it_support_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_help_for_remote_access_labor_pc(ctx: RunContext) -> str:
    """
    Hier können Studenten die nötigen Informationen beziehen, sofern diese einen Remotezugriff auf ein Laborrechner durchführen wollen mittels splashtop.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        str: returns a str with all necessary information's.
    """

    return await get_computerlabore()


# 9)

@support_services_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_kompetenzzentrum_informationen(ctx: RunContext, einrichtung: str) -> list[dict] | str:
    """
    Gives an overview of the selected facility. However, it must be specified. A list of possible elements is returned as soon as the respective facility has been called up. Enter one of the following options to find out what can be found behind it:
    - "innovations-kompetenzzentrum-kuenstliche-intelligenz",
    - "kompetenzzentrum-digitale-lehre",
    - "kompetenzzentrum-bayern-mittel-osteuropa",
    - "kompetenzzentrum-fuer-gesundheit-im-laendlichen-raum",
    - "kompetenzzentrum-grundlagen-ccg",
    - "kompetenzzentrum-fuer-kraft-waerme-kopplung"

    Then call up get_specialized_institution_information with the respective string to get more information.

    Args:
        einrichtung (): Is needed to get an overview
        ctx (RunContext): The current runtime context.

    Returns:
        list[dict] | str: returns a list of dictionaries with the possible specializations
    """
    try:
        return await get_einrichtung_menu_items(kategorie=einrichtung)
    except:
        return ("Only one of the following elements is allowed: innovations-kompetenzzentrum-kuenstliche-intelligenz "
                "kompetenzzentrum-digitale-lehre kompetenzzentrum-bayern-mittel-osteuropa "
                "kompetenzzentrum-fuer-gesundheit-im-laendlichen-raum kompetenzzentrum-grundlagen-ccg "
                "kompetenzzentrum-fuer-kraft-waerme-kopplung")


@support_services_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_specialized_institution_information(ctx: RunContext, specialization: dict[str, str]) -> str:
    """
    Um genauere Informationen über irgendein Kompetenzzentrum zu bekommen kann dieses Werkzeug aufgerufen werden. Um jedoch herauszufinden welche specialization für welches Kompetenzzentrum erlaubt sind muss zuerst: get_kompetenzzentrum_informationen aufgerufen werden.
    Important: The specialization parameter needs following structure: {'title': '<title>', 'url': '<url>'}
    Then call up specialized_institution with the respective dictionary to get more information.

    Args:
        specialization (): Is needed to get an specialization
        ctx (RunContext): The current runtime context.

    Returns:
        str: dictionary of the dates of the events
    """
    try:
        return await get_webpage_as_markdown(specialization["url"])
    except:
        return "Error, the specialization parameter needs following structure: {'title': '<title>', 'url': '<url>}"


@support_services_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_schwarzes_Brett_infos_fuer_studenten(ctx: RunContext) -> list[dict]:
    """
    Nehme nur die aktuelle erste Seite vom schwarzen Brett. Das schwarze Brett ist die erste Anlaufstelle, wenn ein Student nach aktuellen Ereignissen wie Stundenentfall, Ankündigungen oder sonstiges sucht.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        list[dict]: returns a list[dict] with all necessary information's.
    """

    return await oth_schwarzes_brett()


@support_services_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_help_for_teaching(ctx: RunContext) -> list[dict]:
    """
    Falls ein junger Professor/in Hilfe bei der Lehre braucht kann die Person hier die ersten Anlaufstellen finden.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        list[dict]: returns a list[dict] with all necessary information's.
    """

    return await OTH_Support_for_teaching()


@support_services_agent.tool(docstring_format='google', require_parameter_descriptions=True)
@log_tool
async def get_help_for_creating_video(ctx: RunContext) -> str:
    """
    Gibt allgemeine Tips und auch Werkzeuge zurück die man braucht um ein Video professionell zu erstellen.

    Args:
        ctx (RunContext): The current runtime context.

    Returns:
        str: returns a str with all necessary information's.
    """

    return await get_videoproduktion()

