from duckduckgo_search import DDGS
from pydantic import BaseModel, Field
from typing import List, Tuple, Any, Dict, LiteralString
import inspect
import os
import json
import re
from datetime import datetime, timedelta
import aiohttp
import ssl
import certifi

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup, Tag
import time
from docling.document_converter import DocumentConverter
import tempfile

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import aiohttp
import asyncio

import logging

logger = logging.getLogger("tools")


# Settings
class Sources(BaseModel):
    headline: List[str] = Field(default_factory=list)
    link: List[str] = Field(default_factory=list)
    raw_content: List[str] = Field(default_factory=list)
    content: List[str] = Field(default_factory=list)


# Set Firefox to run headless
options = Options()
options.add_argument("--headless")

# cache settings
CACHE_FILE = "cache.json"
CACHE_DURATION = timedelta(weeks=1)


# Allgemeine Funktionen

def track_function():
    current_function = inspect.stack()[1].function
    print(f"Current function: {current_function}")


def track_tool():
    current_function = inspect.stack()[1].function
    print(f"Current tool: {current_function}")


async def load_cache() -> dict:
    """Loads the cache from the JSON file."""
    logger.debug("Function 'load_cache()' called.")
    track_function()
    logger.info(f"Cache file '{CACHE_FILE}' found. Attempting to load.")
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
            logger.info("Cache loaded successfully.")
            return cache_data
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in '{CACHE_FILE}'. Cache will be cleared.", exc_info=True)
        return {}


def save_cache(cache: dict) -> None:
    """Saves the cache to the JSON file."""
    logger.debug("Function 'save_cache()' called.")
    try:
        track_function()
    except NameError:
        logger.warning("Function 'track_function()' not found.")
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=4)
        logger.info(f"Cache saved successfully to '{CACHE_FILE}'.")
    except Exception as e:
        logger.error(f"Failed to save cache to '{CACHE_FILE}': {e}", exc_info=True)


async def call_browser(url: str, use_cache: bool = True) -> BeautifulSoup:
    logger.debug(f"Function 'call_browser()' called with URL: {url}")
    track_function()

    cache = await load_cache()
    # Cache check
    if use_cache:
        if url in cache:
            cached_entry = cache[url]
            cached_date = datetime.fromisoformat(cached_entry["date"])
            if datetime.now() - cached_date < CACHE_DURATION:
                logger.info(f"Using cached data for: {url}")
                html_content = cached_entry["html"]
                return BeautifulSoup(html_content, "html.parser")
            else:
                logger.debug(f"Cache for {url} expired. Fetching fresh data.")

    # Launch browser
    logger.debug("Initializing browser instance.")
    browser_inter = webdriver.Firefox(options=options)
    try:
        logger.info(f"Navigating to URL: {url}")
        browser_inter.get(url)
        time.sleep(2)
        page_source = browser_inter.page_source

        # Update cache
        cache[url] = {
            "html": page_source,
            "date": datetime.now().isoformat()
        }
        logger.info(f"Updating cache with data from: {url}")
        save_cache(cache)
        logger.info(f"Cache saved successfully after fetching: {url}")

        return BeautifulSoup(page_source, "html.parser")
    except Exception as e:
        logger.error(f"Failed to process URL '{url}': {e}", exc_info=True)
        raise e
    finally:
        browser_inter.quit()
        logger.debug(f"Browser instance closed for URL: {url}")


## Functions

def duckduckgo_search(query, max_results=5):
    logger.debug(f"Function 'duckduckgo_search()' called with query: '{query}'")
    track_function()
    try:
        with DDGS() as ddgs:
            results = [result for result in ddgs.text(query, max_results=max_results)]
            logger.info(f"Found {len(results)} results for query: '{query}'")
            return results
    except Exception as e:
        logger.error(f"Failed to search for query '{query}': {e}", exc_info=True)
        return []


def extract_html(sources: Sources) -> Sources:
    logger.debug("Function 'extract_html()' called.")
    track_function()
    try:
        logger.info("Initializing headless Firefox browser.")
        browser = webdriver.Firefox(options=options)
        for idx, search_url in enumerate(sources.link):
            logger.debug(f"Processing URL {idx + 1}/{len(sources.link)}: {search_url}")
            try:
                browser.get(search_url)
                logger.info(f"Navigated to: {search_url}")
                time.sleep(2)
                html = browser.page_source
                sources.raw_content.append(html)
                logger.debug(f"Added HTML content for: {search_url}")
            except Exception as e:
                logger.warning(f"Failed to load URL '{search_url}': {str(e)[:50]}", exc_info=False)
        logger.info("All URLs processed. Closing browser.")
        browser.quit()
        return sources
    except Exception as e:
        logger.error(f"Critical error in 'extract_html()': {e}", exc_info=True)
        browser.quit()
        return sources


# Regex für das Extrahieren der Links
def extract_links(text: str) -> List[str]:
    logger.debug(f"Function 'extract_links()' called with text: {text[:25]}...")
    track_function()
    url_pattern = r"https?://[^\s\"'()>]+(?=\))?"
    links = re.findall(url_pattern, text)
    logger.info(f"Found {len(links)} links in text.")
    return links


# Asynchrone Funktion zum Überprüfen des Links
async def check_page(session, url: str) -> dict:
    track_function()
    result = {"url": url, "valid": True}
    try:
        async with session.get(url, timeout=10) as response:
            logger.debug(f"Received response for {url}: status {response.status}")
            if response.status == 200:
                logger.info(f"URL '{url}' responded with 200 OK")

                content_type = response.headers.get("Content-Type", "").lower()
                logger.debug(f"Content-Type of '{url}': {content_type}")

                if "text/html" in content_type:
                    try:
                        text = await response.text()
                        if "Seite nicht gefunden" in text or "nicht gefunden" in text:
                            result["valid"] = False
                            logger.warning(
                                f"URL '{url}' returned 200 but contains 'not found' message"
                            )
                        else:
                            logger.info(f"URL '{url}' content is valid")
                    except UnicodeDecodeError as ude:
                        result["valid"] = False
                        result["error"] = "UnicodeDecodeError: " + str(ude)
                        logger.warning(f"Unicode decode error for '{url}'")
                else:
                    # Für PDFs, Word etc. akzeptieren wir den Statuscode 200 ...
                    logger.info(f"URL '{url}' is non-HTML (e.g., PDF/Word) and returned 200 OK")
            else:
                result["valid"] = False
                result["status_code"] = response.status
                logger.error(
                    f"URL '{url}' returned invalid status code: {response.status}"
                )
    except Exception as e:
        result["valid"] = False
        result["error"] = str(e)
        logger.error(
            f"Failed to check URL '{url}': {str(e)}",
            exc_info=True,
        )
    return result


# Asynchrone Funktion zum Überprüfen aller Seiten
async def check_all_pages(all_links: List[str], session: aiohttp.ClientSession) -> tuple[BaseException | Any] | list[
    Any]:
    logger.info(f"Starting check_all_pages() with {len(all_links)} links")
    track_function()
    tasks = [check_page(session, link) for link in all_links]
    logger.debug(f"Created {len(tasks)} tasks for link checks using the provided session")

    try:
        # Der asyncio.gather Teil bleibt exakt gleich
        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Finished checking {len(results)} links")
        return results
    except Exception as e:
        # Dieser Fehler-Handler ist gut, um ihn für den Fall der Fälle zu behalten
        logger.error(f"Async error in check_all_pages(): {e}", exc_info=True)
        return []


# Funktion, die alle Links extrahiert und auf ihre Gültigkeit überprüft
async def check_valid_links(text: str, session: aiohttp.ClientSession) -> str:
    logger.info("Starting check_valid_links()...")
    track_function()
    logger.info("Extracting links from text...")
    all_links = extract_links(text)
    logger.info(f"Extracted {len(all_links)} links")

    logger.info("Checking link validity...")
    valid_links = await check_all_pages(all_links, session)
    logger.info(f"Checked {len(valid_links)} links for validity")

    invalid_count = 0
    for checked_link in valid_links:
        if not checked_link["valid"]:
            invalid_count += 1
            text = text.replace(checked_link["url"], " (Sorry faulty link.)")
            logger.debug(f"Replaced invalid link: {checked_link['url']}")

    logger.info(f"Marked {invalid_count}/{len(valid_links)} invalid links")
    return text


def clean_html(sources: Sources) -> Sources:
    logger.debug(f"Starting clean_html() with {len(sources.raw_content)} items")
    track_function()
    for idx, source in enumerate(sources.raw_content):
        logger.debug(f"Processing HTML content {idx + 1}/{len(sources.raw_content)}")
        soup = BeautifulSoup(source, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        sources.content.append(text)
        logger.info(f"Extracted {len(text.splitlines())} lines of text")
    logger.info(f"Processed {len(sources.content)} texts successfully")
    return sources


def full_duckduckgo_search(query: str, num_results: int = 5) -> Sources:
    logger.info(f"Starting full_duckduckgo_search() for query '{query}' ({num_results} results)")
    track_function()
    search_results = duckduckgo_search(query, num_results)
    logger.info(f"Found {len(search_results)} results for query '{query}'")

    sources = {
        'headline': [],
        'link': []
    }

    for i, result in enumerate(search_results):
        logger.debug(f"Processing result {i + 1}/{len(search_results)}: {result['title'][:50]}...")
        sources["headline"].append(result['title'])
        sources["link"].append(result["href"])

    validated_data = Sources(**sources)
    logger.info(f"Created validated Sources object with {len(validated_data.link)} entries")
    return validated_data


###################################

# FAQ on page: https://www.oth-aw.de/studium/studienangebote/faq/

async def crawl_FAQ(user_question: str, url="https://www.oth-aw.de/studium/studienangebote/faq/") -> List[
    tuple[str, str]]:
    logger.debug(f"Starting FAQ crawl for query: '{user_question}'")
    track_function()

    logger.info(f"Accessing FAQ page: {url}")

    try:
        soup = await call_browser(url)
        logger.debug("Page content retrieved successfully")

        section = soup.find("div", class_="section section-default")
        if not section:
            logger.warning(f"Section 'section-default' not found on FAQ page: {url}")
            return [("None", "None")]

        accordion_items = section.find_all("div", class_="accordion-item")
        logger.info(f"Found {len(accordion_items)} FAQ entries on the page")

        faq_data = []
        for item in accordion_items:
            question = item.find("button", class_="accordion-button")
            answer = item.find("div", class_="accordion-body")

            question_text = question.get_text(strip=True) if question else "No question found."
            answer_text = answer.get_text(strip=True) if answer else "No answer found."

            faq_data.append((question_text, answer_text))
            logger.debug(f"Added FAQ entry: '{question_text[:50]}...'")  # Truncated for brevity

        logger.info(f"Successfully parsed {len(faq_data)} FAQ entries")
        return faq_data

    except Exception as e:
        logger.error(f"Failed to crawl FAQ page: {str(e)}", exc_info=True)
        raise e


###################################
# Facilities Overview

async def crawl_elektrotechnik_medien_informatik_overview() -> List[str]:
    logger.debug("Starting Elektrotechnik Medien und Informatik overview crawl")
    track_function()

    url = "https://www.oth-aw.de/hochschule/fakultaeten/elektrotechnik-medien-und-informatik/ueber-die-fakultaet-emi/"
    logger.info(f"Accessing first page: {url}")

    try:
        soup = await call_browser(url)
        logger.debug("First page content retrieved successfully")

        overview = []

        introduction = soup.find("div", class_="ce-bodytext")
        numbers_facts = soup.find("section", id="zahlen-fakten")
        kompetenz = soup.find("section", id="kompetenzfelder")

        overview.append(introduction.get_text())
        logger.info("Added introduction section text to overview")

        overview.append(numbers_facts.get_text())
        logger.info("Added numbers and facts section text to overview")

        overview.append(kompetenz.get_text())
        logger.info("Added competence fields section text to overview")

        overview.append("Quelle: " + url)
        logger.info("Added source URL to overview")

        # Second page
        url = "https://www.oth-aw.de/hochschule/fakultaeten/elektrotechnik-medien-und-informatik/unsere-studienangebote/"
        logger.info(f"Accessing second page: {url}")
        soup = await call_browser(url)
        logger.debug("Second page content retrieved successfully")

        bachelorstudiengaenge = soup.find("section", id="bachelorstudiengaenge")
        masterstudiengaenge = soup.find("section", id="masterstudiengaenge")
        berufsfelder = soup.find("section", id="berufsfelder")

        overview.append(bachelorstudiengaenge.get_text())
        logger.info("Added bachelor programs section text to overview")

        overview.append(masterstudiengaenge.get_text())
        logger.info("Added master programs section text to overview")

        overview.append(berufsfelder.get_text())
        logger.info("Added career fields section text to overview")

        overview.append("Quelle: " + url)
        logger.info("Added second source URL to overview")

        if not overview:
            logger.warning("No data retrieved from pages; returning default")
            return ["None"]
        logger.info(f"Successfully retrieved {len(overview)} sections")
        return overview

    except Exception as e:
        logger.error(f"Failed to crawl Elektrotechnik Medien und Informatik overview: {str(e)}", exc_info=True)
        raise e


async def crawl_maschinenbau_umwelttechnik_overview() -> List[str]:
    logger.debug("Starting Maschinenbau Umwelttechnik overview crawl")
    track_function()

    url = "https://www.oth-aw.de/hochschule/fakultaeten/maschinenbau-umwelttechnik/fakultaet-mb-ut/"
    logger.info(f"Accessing page: {url}")

    try:
        soup = await call_browser(url)
        logger.debug("Page content retrieved successfully")

        overview = []

        introduction_0 = soup.find("div", id="c38259")
        introduction_1 = soup.find("div", id="c38516")
        studienangebot = soup.find("section", id="studienangebot")
        forschung = soup.find("section", id="forschung")
        labore = soup.find("section", id="labore")
        ansprechpartner = soup.find("section", id="ansprechpartner")

        try:
            overview.append(introduction_0.get_text())
            logger.info("Added introduction_0 section text to overview")
            overview.append(introduction_1.get_text())
            logger.info("Added introduction_1 section text to overview")
            overview.append(studienangebot.get_text())
            logger.info("Added studienangebot section text to overview")
            overview.append(forschung.get_text())
            logger.info("Added forschung section text to overview")
            overview.append(labore.get_text())
            logger.info("Added labore section text to overview")
            overview.append(ansprechpartner.get_text(separator=" ", strip=True))
            logger.info("Added ansprechpartner section text to overview")

        except AttributeError:
            logger.error("Missing section element while parsing content", exc_info=True)
            raise AttributeError("Missing section element while parsing")

        overview.append("Quelle: " + url)
        logger.info("Added source URL to overview")

        if not overview:
            logger.warning("No data retrieved from page; returning default")
            return ["None"]
        logger.info(f"Successfully retrieved {len(overview)} sections")
        return overview

    except Exception as e:
        logger.error(f"Failed to crawl Maschinenbau Umwelttechnik overview: {str(e)}", exc_info=True)
        raise e


async def crawl_weiden_business_school_overview() -> List[str]:
    logger.debug("Starting Weiden Business School overview crawl")
    track_function()

    url = "https://www.oth-aw.de/hochschule/fakultaeten/weiden-business-school/willkommen-an-der-weiden-business-school/"
    logger.info(f"Accessing page: {url}")

    try:
        soup = await call_browser(url)
        logger.debug("Page content retrieved successfully")

        overview = []

        introduction_0 = soup.find("div", id="c19079")
        unser_leistungsangebot = soup.find("section", id="unser-leistungsangebot")
        zahlen__fakten = soup.find("section", id="zahlen--fakten")
        social_media = soup.find("section", id="social-media")

        try:
            overview.append(introduction_0.get_text())
            logger.info("Added introduction section text to overview")
            overview.append(unser_leistungsangebot.get_text())
            logger.info("Added 'unser-leistungsangebot' section text to overview")
            overview.append(zahlen__fakten.get_text())
            logger.info("Added 'zahlen--fakten' section text to overview")
            social_media_text = social_media.get_text() + "Facebook: https://www.facebook.com/OTHAmbergWeiden" + "Instagramm: https://www.instagram.com/othambergweiden/"
            overview.append(social_media_text)
            logger.info("Added social media section text to overview")

        except AttributeError:
            logger.error("Missing section element while parsing content", exc_info=True)
            raise AttributeError("Missing section element while parsing")

        overview.append("Quelle: " + url)
        logger.info("Added source URL to overview")

        if not overview:
            logger.warning("No data retrieved from page; returning default")
            return ["None"]
        logger.info(f"Successfully retrieved {len(overview)} sections")
        return overview

    except Exception as e:
        logger.error(f"Failed to crawl Weiden Business School overview: {str(e)}", exc_info=True)
        raise e


async def crawl_wirtschaft_gesundheit_overview() -> List[str]:
    logger.debug("Starting Wirtschaft und Gesundheit overview crawl")
    track_function()

    url = "https://www.oth-aw.de/hochschule/fakultaeten/wirtschaftsingenieurwesen-und-gesundheit/ueber-die-fakultaet-wirtschaftsingenieurwesen-und-gesundheit/"
    logger.info(f"Accessing first page: {url}")

    try:
        soup = await call_browser(url)
        logger.debug("First page content retrieved successfully")

        overview = []

        unser_leistungsangebot = soup.find("section", id="unser-leistungsangebot")
        zahlen_fakten = soup.find("section", id="zahlen-fakten")

        try:
            overview.append(unser_leistungsangebot.get_text())
            logger.info("Added 'unser-leistungsangebot' section text to overview")
            overview.append(zahlen_fakten.get_text())
            logger.info("Added 'zahlen-fakten' section text to overview")
        except AttributeError:
            logger.error("Missing section element while parsing first page content", exc_info=True)
            raise AttributeError("Missing section element while parsing first page")

        overview.append("Quelle: " + url)
        logger.info("Added first source URL to overview")

        # Second page
        url = "https://www.oth-aw.de/hochschule/fakultaeten/wirtschaftsingenieurwesen-und-gesundheit/unsere-studienangebote/"
        logger.info(f"Accessing second page: {url}")
        soup = await call_browser(url)
        logger.debug("Second page content retrieved successfully")

        berufsfelder = soup.find("section", id="berufsfelder")
        bachelorstudiengaenge = soup.find("section", id="bachelorstudiengaenge")
        masterstudiengaenge = soup.find("section", id="masterstudiengaenge")
        weiterbildungsmaster = soup.find("section", id="weiterbildungsmaster")
        zertifizierungskurse = soup.find("section", id="zertifizierungskurse")

        try:
            overview.append(berufsfelder.get_text())
            logger.info("Added 'berufsfelder' section text to overview")
            overview.append(bachelorstudiengaenge.get_text())
            logger.info("Added 'bachelorstudiengaenge' section text to overview")
            overview.append(masterstudiengaenge.get_text())
            logger.info("Added 'masterstudiengaenge' section text to overview")
            overview.append(weiterbildungsmaster.get_text())
            logger.info("Added 'weiterbildungsmaster' section text to overview")
            overview.append(zertifizierungskurse.get_text())
            logger.info("Added 'zertifizierungskurse' section text to overview")
        except AttributeError:
            logger.error("Missing section element while parsing second page content", exc_info=True)
            raise AttributeError("Missing section element while parsing second page")

        overview.append("Quelle: " + url)
        logger.info("Added second source URL to overview")

        if not overview:
            logger.warning("No data retrieved from pages; returning default")
            return ["None"]
        logger.info(f"Successfully retrieved {len(overview)} sections")
        return overview

    except Exception as e:
        logger.error(f"Failed to crawl Wirtschaft und Gesundheit overview: {str(e)}", exc_info=True)
        raise e


###################################
# Stundenpläne

async def find_similar_sentence(data: List[Tuple[str, str]], query: str, top_n: int = 2) -> List[Tuple[str, str, float]]:
    logger.debug(f"Finding similar sentences for query '{query}' with top_n={top_n}")
    track_function()

    course_list = [course for course, _ in data]
    logger.info(f"Processed {len(course_list)} courses for similarity analysis")

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(course_list + [query])
    logger.debug(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    query_vector = tfidf_matrix[-1]
    logger.debug(f"Query vector shape: {query_vector.shape}")

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix[:-1]).flatten()
    logger.debug(f"Similarity scores shape: {similarity_scores.shape}")

    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    logger.debug(f"Top {top_n} indices selected: {top_indices}")

    logger.info(
        f"Found {len(top_indices)} similar sentences with average similarity: {similarity_scores[top_indices].mean():.2f}")
    return [(data[i][0], data[i][1], similarity_scores[i]) for i in top_indices]


async def crawl_stundenplan() -> List[Tuple[str, str]]:
    logger.debug("Starting crawl_stundenplan()...")
    track_function()

    url = "https://www.oth-aw.de/studium/im-studium/organisatorisches/stunden-und-pruefungsplaene/"
    logger.info(f"Accessing URL: {url}")

    try:
        soup = await call_browser(url)
        logger.debug("Page content retrieved successfully")

        plaene = soup.find("div", class_="tab-content")
        logger.debug(f"Found 'tab-content' section: {plaene is not None}")

        stundenplan_links = []
        for a_tag in plaene.find_all("a", title=True):
            if "Stundenplan" in a_tag["title"]:
                link = "https://www.oth-aw.de/" + a_tag["href"]
                course = a_tag["title"].replace("Stundenplan ", "").strip()
                stundenplan_links.append((course, link))
                logger.debug(f"Added course: '{course[:50]}' with link: {link}")

        stundenplan_links.append(("Quelle:", url))
        logger.info(f"Collected {len(stundenplan_links) - 1} course schedule links")

        if not stundenplan_links:
            logger.warning("No valid course schedule links found; returning default")
            return [("None", "None")]
        logger.info(f"Returning {len(stundenplan_links) - 1} valid links")
        return stundenplan_links

    except Exception as e:
        logger.error(f"Failed to crawl Stundenplan: {str(e)}", exc_info=True)
        raise e


async def crawl_pruefungsplaene() -> List[Tuple[str, str]]:
    logger.debug("Starting crawl_pruefungsplaene()...")
    track_function()

    url = "https://www.oth-aw.de/studium/im-studium/organisatorisches/stunden-und-pruefungsplaene/"
    logger.info(f"Accessing URL: {url}")

    try:
        soup = await call_browser(url)
        logger.debug("Page content retrieved successfully")

        plaene = soup.find("div", class_="tab-content")
        logger.debug(f"Found 'tab-content' section: {plaene is not None}")

        pruefungsplan_links = []
        for a_tag in plaene.find_all("a", title=True):
            if "Prüfungsplan" in a_tag["title"]:
                link = "https://www.oth-aw.de/" + a_tag["href"]
                course = a_tag["title"].replace("Prüfungsplan ", "").strip()
                pruefungsplan_links.append((course, link))
                logger.debug(f"Added course: '{course[:50]}' with link: {link}")

        pruefungsplan_links.append(("Quelle:", url))
        logger.info(f"Collected {len(pruefungsplan_links) - 1} exam schedule links")

        if not pruefungsplan_links:  # Exclude the source entry
            logger.warning("No valid exam schedule links found; returning default")
            return [("None", "None")]
        logger.info(f"Returning {len(pruefungsplan_links) - 1} valid links")
        return pruefungsplan_links

    except Exception as e:
        logger.error(f"Failed to crawl Prüfungspläne: {str(e)}", exc_info=True)
        raise e


###################################
# Fragen zum Studienablauf
async def beurlaubung() -> List[str]:
    logger.debug("Starting beurlaubung()...")
    track_function()

    url = "https://www.oth-aw.de/studium/im-studium/organisatorisches/studienablauf/"
    logger.info(f"Accessing URL: {url}")

    try:
        soup = await call_browser(url)
        logger.debug("Page content retrieved successfully")

        Beurlaubung = soup.find("section", id="beurlaubung")
        if not Beurlaubung:
            logger.error(f"Section 'beurlaubung' not found on page: {url}")
            raise AttributeError

        overview = [Beurlaubung.get_text(separator=" ", strip=True)]
        logger.info("Added beurlaubung section text to overview")
        overview.append("Source: " + url)

        logger.info(f"Successfully retrieved {len(overview)} entries")
        if not overview:
            return ["None"]
        return overview

    except Exception as e:
        logger.error(f"Failed to crawl beurlaubung: {str(e)}", exc_info=True)
        raise e


async def studiengangswechsel() -> List[str]:
    logger.debug("Starting studiengangswechsel()...")
    track_function()

    url = "https://www.oth-aw.de/studium/im-studium/organisatorisches/studienablauf/"
    logger.info(f"Accessing URL: {url}")

    try:
        soup = await call_browser(url)
        logger.debug("Page content retrieved successfully")

        studiengangswechsel = soup.find("section", id="studiengangwechsel")
        if not studiengangswechsel:
            logger.error(f"Section 'studiengangwechsel' not found on page: {url}")
            raise AttributeError

        overview = [studiengangswechsel.get_text(separator=" ", strip=True)]
        logger.info("Added studiengangswechsel section text to overview")
        overview.append("Source: " + url)

        logger.info(f"Successfully retrieved {len(overview)} entries")
        if not overview:
            return ["None"]
        return overview

    except Exception as e:
        logger.error(f"Failed to crawl studiengangswechsel: {str(e)}", exc_info=True)
        raise e


async def rueckmeldungen_studentenwerksbeitrag() -> List[str]:
    logger.debug("Starting rueckmeldungen_studentenwerksbeitrag()...")
    track_function()

    url = "https://www.oth-aw.de/studium/im-studium/organisatorisches/studienablauf/"
    logger.info(f"Accessing URL: {url}")

    try:
        soup = await call_browser(url)
        logger.debug("Page content retrieved successfully")

        ru_st = soup.find("section", id="rueckmeldung--studentenwerksbeitrag")
        if not ru_st:
            logger.error(f"Section 'rueckmeldung--studentenwerksbeitrag' not found on page: {url}")
            raise AttributeError

        overview = [ru_st.get_text(separator=" ", strip=True)]
        logger.info("Added rueckmeldung section text to overview")
        overview.append("Source: " + url)

        logger.info(f"Successfully retrieved {len(overview)} entries")
        if not overview:
            return ["None"]
        return overview

    except Exception as e:
        logger.error(f"Failed to crawl rueckmeldungen: {str(e)}", exc_info=True)
        return ["Error: ", str(e)]


async def exmatrikulation() -> List[str]:
    logger.debug("Starting exmatrikulation()...")
    track_function()

    url = "https://www.oth-aw.de/studium/im-studium/organisatorisches/studienablauf/"
    logger.info(f"Accessing URL: {url}")

    try:
        soup = await call_browser(url)
        logger.debug("Page content retrieved successfully")

        exmatrikulation = soup.find("section", id="exmatrikulation")
        if not exmatrikulation:
            logger.error(f"Section 'exmatrikulation' not found on page: {url}")
            raise AttributeError

        overview = [exmatrikulation.get_text(separator=" ", strip=True)]
        logger.info("Added exmatrikulation section text to overview")

        for a_tag in exmatrikulation.find_all("a"):
            note = a_tag.get_text(strip=True)
            link = "https://www.oth-aw.de" + a_tag["href"]
            overview.append(f"{note}: {link}")
            logger.debug(f"Added link: {note} -> {link}")

        overview.append("Source: " + url)
        logger.info(f"Successfully retrieved {len(overview)} entries")
        if not overview:
            return ["None"]
        return overview

    except Exception as e:
        logger.error(f"Failed to crawl exmatrikulation: {str(e)}", exc_info=True)
        raise e


async def abschlussarbeiten() -> List[str]:
    logger.debug("Starting abschlussarbeiten()...")
    track_function()

    url = "https://www.oth-aw.de/studium/im-studium/organisatorisches/studienablauf/"
    logger.info(f"Accessing URL: {url}")

    try:
        soup = await call_browser(url)
        logger.debug("Page content retrieved successfully")

        Abschlussarbeiten = soup.find("section", id="abschlussarbeiten")
        if not Abschlussarbeiten:
            logger.error(f"Section 'abschlussarbeiten' not found on page: {url}")
            raise AttributeError

        overview = [Abschlussarbeiten.get_text(separator=" ", strip=True)]
        logger.info("Added abschlussarbeiten section text to overview")

        for a_tag in Abschlussarbeiten.find_all("a"):
            note = a_tag.get_text(separator=" ", strip=True)
            link = "https://www.oth-aw.de" + a_tag["href"]
            if "EMI" in a_tag["href"]:
                note += " (EMI)"
            elif "MBUT" in a_tag["href"]:
                note += " (MBUT)"
            overview.append(f"{note}: {link}")
            logger.debug(f"Added link: {note} -> {link}")

        overview.append("Source: " + url)
        logger.info(f"Successfully retrieved {len(overview)} entries")

        if not overview:
            return ["None"]

        return overview

    except Exception as e:
        logger.error(f"Failed to crawl abschlussarbeiten: {str(e)}", exc_info=True)
        raise e


async def praxissemester() -> List[str]:
    logger.debug("Starting praxissemester()...")
    track_function()

    url = "https://www.oth-aw.de/studium/im-studium/organisatorisches/studienablauf/"
    logger.info(f"Accessing URL: {url}")

    try:
        soup = await call_browser(url)
        logger.debug("Page content retrieved successfully")

        Praxissemester = soup.find("section", id="praxissemester--praxisphasen")
        if not Praxissemester:
            logger.error(f"Section 'praxissemester--praxisphasen' not found on page: {url}")
            raise AttributeError

        table = soup.find("table")
        logger.debug(f"Table found: {'Yes' if table else 'No'}")
        if not table:
            logger.error(f"Table not found on page: {url}")
            raise AttributeError

        headers = [th.get_text(strip=True) for th in Praxissemester.find_all("th")]
        logger.debug(f"Extracted headers: {headers}")

        rows = table.find_all("tr")[1:]  # Skip header row
        logger.info(f"Parsed {len(rows)} rows from table")

        data = {headers[0]: [], headers[1]: []}

        for row in rows:
            cells = row.find_all("td")
            if len(cells) == 2:
                data[headers[0]].append(cells[0].get_text("\n", strip=True))
                data[headers[1]].append(cells[1].get_text("\n", strip=True))
                logger.debug(f"Processed row: {cells[0].text[:50]}... | {cells[1].text[:50]}...")

        overview = []
        try:
            overview.append(str(data))
            logger.info("Added table data to overview")
            overview.append("Source: " + url)
            logger.info("Added source URL to overview")

            for a_tag in Praxissemester.find_all("a"):
                note = a_tag.get_text(separator=" ", strip=True)
                link = a_tag["href"]
                if "oth-aw.de" not in link and "primuss.de" not in link and "www" not in link:
                    link = "https://www.oth-aw.de" + link
                overview.append(f"{note}: {link}")
                logger.debug(f"Added link: '{note[:50]}' -> {link}")

        except AttributeError:
            logger.error("Error processing links or data", exc_info=True)
            raise AttributeError("Error processing links or data")

        if not overview:
            logger.warning("No valid data found; returning default")
            return ["None"]
        logger.info(f"Successfully retrieved {len(overview)} entries")
        return overview

    except Exception as e:
        logger.error(f"Failed to crawl praxissemester: {str(e)}", exc_info=True)
        raise e


###################################
# Überblick über alle Studiengänge

async def all_study_programs(append=True) -> List[Tuple[str, str]]:
    logger.debug("Starting all_study_programs()...")
    track_function()

    url = "https://www.oth-aw.de/studium/studienangebote/studiengaenge/"
    logger.info(f"Accessing URL: {url}")

    try:
        soup = await call_browser(url)
        print("blublablabbbb")
        logger.debug("Page content retrieved successfully")

        plaene = soup.find("div", class_="in2studyfinder__results")
        if not plaene:
            logger.error(f"Critical 'in2studyfinder__results' section not found on page: {url}")
            return [("None", "None")]

        study_programs = []
        for a_tag in plaene.find_all("a"):
            link = "https://www.oth-aw.de" + a_tag["href"]
            course = a_tag.get_text(strip=True)

            if append:
                if "master" in link.lower():
                    course += " (Master)"
                elif "bachelor" in link.lower():
                    course += " (Bachelor)"

            study_programs.append((course, link))
            logger.debug(f"Added course: '{course[:50]}' with link: {link}")

        study_programs.append(("Alle Studiengänge sind hier zu finden:", url))
        logger.info(f"Collected {len(study_programs) - 1} valid study programs")

        if not study_programs:  # Exclude the source entry
            logger.warning("No valid study programs found; returning default")
            return [("None", "None")]
        logger.info(f"Returning {len(study_programs) - 1} valid entries")
        return study_programs

    except Exception as e:
        logger.error(f"Failed to crawl study programs: {str(e)}", exc_info=True)
        raise e


async def get_more_details_about_specific_study_program(study_program: str, bachelor: bool = True) -> str:
    logger.debug(f"Starting get_more_details() for query: '{study_program}' (bachelor={bachelor})")
    track_function()

    try:
        everything = await all_study_programs(append=False)
        logger.info(f"Retrieved {len(everything)} study programs from all_study_programs()")

        if len(everything) > 2:
            logger.info(f"Finding similar sentences for query: '{study_program}'")
            results = await find_similar_sentence(data=everything, query=study_program)
            logger.debug(f"Similar results: {results[:3]} (truncated)")

            url = "https://www.oth-aw.de"
            for i, result in enumerate(results):
                if "master" in result[1].lower() and not bachelor:
                    url = result[1]
                elif "bachelor" in result[1].lower() and bachelor:
                    url = result[1]
                else:
                    try:
                        url = result[1]
                    except Exception as e:
                        logger.warning(f"Failed to process result {i}: {str(e)}")
                        url = "https://www.oth-aw.de/studium/studienangebote/studiengaenge/"

            logger.info(f"Selected URL: {url}")
            try:
                soup = await call_browser(url)
                logger.debug("Page content retrieved successfully")
                info = soup.find("main", role="main")
                try:
                    only_text = info.get_text(separator=" ", strip=True)
                except AttributeError:
                    logger.error(f"Website crawl has an Error with this url: {url}")
                    return f"No information found! Error with this study_program: {study_program}"
                logger.info(f"Extracted {len(only_text.splitlines())} lines of text")
                logger.info(f"Extracted text: {only_text}")
                return only_text
            except Exception as e:
                logger.error(f"Failed to process URL '{url}': {str(e)}", exc_info=True)
                raise e
        else:
            logger.warning("Too few study programs retrieved; returning error")
            return "Error while searching for the study programs."

    except Exception as e:
        logger.error(f"Critical error in get_more_details(): {str(e)}", exc_info=True)
        raise e


###################################
# Team / Professoren

def get_team(search_term: str) -> List[dict]:
    logger.debug(f"Starting get_team() for search_term: '{search_term}'")
    track_function()

    url = "https://www.oth-aw.de/hochschule/ueber-uns/personen/"
    logger.info(f"Accessing URL: {url}")

    try:
        driver = webdriver.Firefox(options=options)
        logger.debug("Firefox browser initialized")
        driver.get(url)
        logger.debug("Page loaded successfully")

        login_hinweis = "Bitte melden Sie sich mit Ihren Zugangsdaten"
        if login_hinweis in driver.page_source:
            logger.warning("Anmeldeseite erkannt. Die Suche wird nicht durchgeführt.")
            return [{"Error": f"It looks like there is now a login screen. Unfortunately I can't access the data for you, but here is the link:", "link": url}]

        try:
            # Warte bis zu 10 Sekunden, bis das Element mit der ID "searchword" vorhanden UND klickbar ist.
            wait = WebDriverWait(driver, 10)
            search_input = wait.until(EC.element_to_be_clickable((By.ID, "searchword")))
            search_input.send_keys(search_term)
            logger.info(f"Entered search term: '{search_term}'")

        except TimeoutException:
            logger.error("Das Suchfeld 'searchword' wurde nach 10 Sekunden nicht gefunden oder war nicht klickbar.")
            return [{"Error": f"Could not search for the given request, please take a look at the <insert_link>", "link": url}]

        soup = BeautifulSoup(driver.page_source, "html.parser")
        logger.debug("Page source parsed to BeautifulSoup")

        table = soup.find("table", {"id": "mitarbeiter"})
        if not table:
            logger.error("Table 'mitarbeiter' not found on page")
            return [{"Error": f"Could not found the requested information, please take a look at the <insert_link>", "link": url}]

        rows = table.find("tbody").find_all("tr")
        logger.info(f"Parsed {len(rows)} rows from table")

        data = []
        for row in rows:
            cols = row.find_all("td")

            # Skip empty/error rows
            if len(cols) == 1 and "No matching" in cols[0].text:
                logger.debug("Skipping 'No matching records' row")
                continue

            name_td = cols[0]
            name_link = name_td.find("a")
            name_text = name_link.text.strip() if name_link else name_td.text.strip()
            name_href = name_link["href"] if name_link and "href" in name_link.attrs else ""
            data_search = name_td.get("data-search", "").strip()

            role_td = cols[1]
            role_link = role_td.find("a")
            role_text = role_link.text.strip() if role_link else role_td.text.strip()
            role_href = role_link["href"] if role_link and "href" in role_link.attrs else ""

            lehrgebiet = cols[2].text.strip()

            entry = {
                "Name": name_text,
                "Titel/Tel": data_search,
                "Profil-URL": "https://www.oth-aw.de" + name_href,
                "Rolle": role_text,
                "Rollen-URL": "https://www.oth-aw.de" + role_href,
                "Lehrgebiet": lehrgebiet
            }
            data.append(entry)
            logger.debug(f"Added entry for: {entry['Name']}")

        logger.info(f"Collected {len(data)} entries")
        return data

    except Exception as e:
        logger.error(f"Failed to retrieve team data: {str(e)}", exc_info=True)
        raise e

    finally:
        if driver:
            driver.close()
            logger.debug("Firefox browser closed")


async def extract_links_with_titles(profile_url: str) -> List[Tuple[str, str]]:
    logger.debug(f"Starting extract_links_with_titles() for URL: '{profile_url}'")
    track_function()

    try:
        soup = await call_browser(url=profile_url)
        logger.debug("Page content retrieved successfully")

        side_section = soup.find("div", class_="section-column subnav-wrap")
        if not side_section:
            logger.warning("Section 'section-column subnav-wrap' not found")
            return []

        links = side_section.find_all("a", href=True, title=True)
        logger.info(f"Found {len(links)} links in section")

        extracted_links = []
        for link in links:
            href = link["href"]
            title = link["title"]
            if "#" not in href:
                full_url = "https://www.oth-aw.de" + href
                extracted_links.append((full_url, title))
                logger.debug(f"Added link: {title[:50]} -> {full_url}")

        logger.info(f"Extracted {len(extracted_links)} valid links")
        return extracted_links

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{profile_url}': {str(e)}", exc_info=True)
        return []


###################################
# Essenziell für den Studienalltag  ---- Ab hier noch nicht in tools umgewandelt!!!


async def get_wifi_eduroam_vpn_info_live() -> str:
    url = "https://www.oth-aw.de/hochschule/services/online-services/wlan-vpn-netzwerk/"

    try:
        soup = await call_browser(url=url)
        side_section = soup.find("div", class_="section section-default", id="main")

        info = f"""Um den VPN-Client herunterzuladen müssen sie sich der Webseite anmelden. Dies kann ich leider nicht für dich machen 😥. Aber hier ist der link: {url}\n\n\n"""

        wifi_ = side_section.find("div", id="c38166")
        eduroam_ = side_section.find("div", id="c39437")
        vpn_zugang_ = side_section.find("div", id="c39435")
        forti_client_ = side_section.find("div", id="c39440")

        return info + wifi_.get_text() + eduroam_.get_text() + vpn_zugang_.get_text() + forti_client_.get_text()

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


async def get_email_contact_calender_live() -> str:
    url = "https://www.oth-aw.de/hochschule/services/online-services/e-mail-kontakte-kalender/"

    try:
        soup = await call_browser(url=url)
        side_section = soup.find("div", class_="section section-default", id="main")

        info = f"""Hier müsste man sich anmelden um zusätzliche Funktionen freizuschalten. Dies kann ich leider nicht für dich machen 😥. Aber hier ist der link: {url}\n\n\n"""

        overview_ = side_section.find("div", id="c38199")
        webmail_ = side_section.find("div", id="c38210")
        microfocus_groupwise_ = side_section.find("div", id="c38200")  # Davor einfügen das man sich anmelden muss
        other_webmail_ = side_section.find("div", id="c38203")
        sync_mail_ = side_section.find("div", id="c38204")
        sync_mail_1_ = side_section.find("div", id="c38186")
        sync_mail_2_ = side_section.find("div", id="c38206")  # Davor einfügen das man sich anmelden muss
        email_verteiler_ = side_section.find("div", id="c38214")

        return overview_.get_text() + webmail_.get_text() + microfocus_groupwise_.get_text() + info + other_webmail_.get_text() + sync_mail_.get_text() + sync_mail_1_.get_text() + sync_mail_2_.get_text() + info + email_verteiler_.get_text()

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


async def get_file_exchange() -> str:
    url = "https://www.oth-aw.de/hochschule/services/online-services/dateizugriff-dateiaustausch/"

    try:
        soup = await call_browser(url=url)
        side_section = soup.find("div", class_="section section-default", id="main")

        info = f"""Es gibt hier einen Bereich auf welchen ich nicht zugreifen kann. Da musst du dich selber einloggen falls die Antwort nicht ausreicht. Link: {url}"""

        myfiles_ = side_section.find("div", id="c38216")
        gigamove_ = side_section.find("div", id="c38218")
        tip = (
            "\n\nTipp: Die simpelste Methode ist es den myfiles server der OTH zu verwenden. Dazu gehe einfach auf den "
            "folgenden Link: https://myfiles.oth-aw.de/filr/login")

        return info + myfiles_.get_text() + gigamove_.get_text() + tip

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


async def get_print_info() -> str:
    url = "https://www.oth-aw.de/hochschule/services/online-services/drucken-fuer-studierende/"

    try:
        soup = await call_browser(url=url)
        side_section = soup.find("div", class_="section section-default", id="main")

        overview_ = "Drucken fuer studierende \n\n"
        system_papercut_ = side_section.find("div", id="c38284").get_text()
        print_anywhere_ = side_section.find("div", id="c38286").get_text()
        rz_pool_ = side_section.find("div", id="c38288").get_text()
        auffuellen_ = side_section.find("div", id="c38292").get_text()
        standort_ = side_section.find("div", id="c2009").get_text()
        more_infos_ = side_section.find("div", id="c2010").get_text()
        online_print_ = side_section.find("div", id="c38294").get_text()

        return overview_ + "\n" + system_papercut_ + "\n" + print_anywhere_ + "\n" + rz_pool_ + "\n" + auffuellen_ + "\n" + standort_ + "\n" + more_infos_ + "\n" + online_print_

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


def lageplan_raumfinder() -> List[str]:
    plan_weiden = "Lageplan Weiden: https://www.oth-aw.de/files/oth-aw/Standorte/Lageplan_OTH-AW_Weiden.pdf"
    plan_amberg = "Lageplan Amberg: https://www.oth-aw.de/files/oth-aw/Standorte/Lageplan_OTH-AW_Amberg.pdf"
    info = """Wenn du einen Raum oder ein Gebäude suchst dann ist dieser Raumfinder das passende Werkzeug für dich: 
    https://www.oth-aw.de/hochschule/services/online-services/raumfinder/ Ich kann die Webseite leider nicht bedienen, 
    aber ich hoffe dir hilft der. 🤗"""
    return [plan_weiden, plan_amberg, info]


async def bibliothek_kontakt():
    url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/ueber-uns/"

    try:
        soup = await call_browser(url=url)
        side_section = soup.find("div", class_="section section-default", id="main")

        kontakt = side_section.find("div", id="c483").get_text()
        return kontakt

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


def _get_headers_from_section(soup, section_id):  # Hilfsfunktion
    section = soup.find("section", id=section_id)
    if not section:
        return []
    headers = section.find_all("h3")
    return [h.get_text(strip=True) for h in headers]


def _get_section_by_header(soup, section_id, header_title):  # Hilfsfunktion
    section = soup.find("section", id=section_id)
    if not section:
        return None

    all_containers = section.find_all("div", class_="frame-group-container")

    for container in all_containers:
        header = container.find("h3")
        if header and header.get_text(strip=True) == header_title:
            return container.get_text()

    return None


async def bibliothek_aktuelles() -> List[str] | str:
    url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/ueber-uns/"

    try:
        soup = await call_browser(url=url)
        headers = _get_headers_from_section(soup, "aktuelle-hinweise")
        return headers

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


async def bibliothek_aktuelles_specific(header: str) -> str:
    """
    Important. Only the headers from the function "bibliothek_aktuelles" are working!!!
    """

    url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/ueber-uns/"
    soup = await call_browser(url=url)

    check_header = bibliothek_aktuelles()
    if header in check_header:
        section_html = _get_section_by_header(soup, "aktuelle-hinweise", header)  # oder ein anderer Titel
        return section_html
    else:
        return f"Only following headers are allowed: {check_header}"


async def _extract_news_items(soup, section_id=""):
    section = soup.find("section", id=section_id) if section_id else soup
    news_items = []

    # Finde alle <li class="list-group-item"> – jeder davon ist ein News-Eintrag
    list_items = section.find_all("li", class_="list-group-item")
    for item in list_items:
        # Der Hauptlink zur News ist in <h2 class="h5"><a>
        headline_link = item.find("h2", class_="h5")
        if headline_link and headline_link.a:
            title = headline_link.a.get_text(strip=True)
            href = headline_link.a.get("href", "").strip()

            news_items.append({
                "title": title,
                "href": "https://www.oth-aw.de" + href,
            })

    return news_items


async def bibliothek_news():
    url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/ueber-uns/"
    soup = await call_browser(url=url)

    news = _extract_news_items(soup)
    return news


async def bibliothek_oeffnungszeiten() -> str:
    url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/ueber-uns/"

    try:
        soup = await call_browser(url=url)
        side_section = soup.find("div", class_="section section-default")

        zeiten = side_section.find("section", id="oeffnungszeiten").get_text()
        return zeiten

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


async def bibliothek_24h_open() -> str:
    url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/ueber-uns/"

    try:
        soup = await call_browser(url=url)
        side_section = soup.find("div", class_="section section-default")

        twentyfour = side_section.find("section", id="24h-bibliothek").get_text()
        return twentyfour

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


async def _bibliothek_team_extend(sparse_information: List[dict[str, str]]) -> str | list[dict[str, str | None | Any]]:
    team_data = []
    urls = [url["profile_link"] for url in sparse_information]
    urls = list(set(urls))
    for url in urls:

        try:
            soup = await call_browser(url=url)
            side_section = soup.find("div", id="c30475")

            # Name extrahieren (h1)
            name_tag = side_section.find('h1')
            name = name_tag.get_text(strip=True) if name_tag else None

            # Position (steht im <p><a> direkt nach dem Namen)
            position_tag = name_tag.find_next('p') if name_tag else None
            position = position_tag.get_text(strip=True) if position_tag else None

            # Telefonnummer, Fax und E-Mail
            contact_block = side_section.find('p', class_='contactblock1')
            phone = fax = email = None
            if contact_block:
                text = contact_block.get_text(separator="\n", strip=True)
                phone_match = re.search(r'Telefon\s+(.+)', text)
                fax_match = re.search(r'Fax\s+(.+)', text)
                phone = phone_match.group(1) if phone_match else None
                fax = fax_match.group(1) if fax_match else None

                # Email zusammensetzen aus sichtbarem Text
                email_tag = contact_block.find('a', href="#")
                if email_tag:
                    email_parts = list(email_tag.stripped_strings)
                    email = ''.join(email_parts)

            # Standort (folgender <p> nach Kontaktblock)
            location_tag = contact_block.find_next('p') if contact_block else None
            location = location_tag.get_text(strip=True) if location_tag else None

            team_data.append({
                "name": name,
                "position": position,
                "telefon": phone,
                "fax": fax,
                "email": email,
                "standort": location,
                "profil-link": url
            })

        except Exception as e:
            logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
            return f"Please check: {url}"

    return team_data


async def bibliothek_team() -> list[dict[str, str | LiteralString | Any]] | str:
    url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/ueber-uns/"

    try:
        soup = await call_browser(url=url)
        side_section = soup.find("div", class_="section section-default")
        people_data = []

        modals = soup.find_all("div", class_="modal fade")

        for modal in modals:
            person = {}

            # Name (steht im <h5> → <a>)
            name_tag = modal.find("h5", class_="modal-title")
            if name_tag and name_tag.a:
                person["name"] = name_tag.a.text.strip()

            # Profil-Link
            profile_link = name_tag.a["href"] if name_tag and name_tag.a else None
            person["profile_link"] = "https://www.oth-aw.de" + profile_link

            people_data.append(person)

        return _bibliothek_team_extend(people_data)

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


async def bibliothek_hinweis_als_externer_nutzer() -> str:
    url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/lernort-bibliothek-und-services/"

    try:
        soup = await call_browser(url=url)
        side_section = soup.find("main", class_="section-column maincontent-wrap", role="main")

        hinweise = side_section.find("section", id="hinweise-fuer-externe-benutzer").get_text()

        info = "\n\nDas ist der Link zu der Einerständniserklärung: https://www.oth-aw.de/files/oth-aw/Einrichtungen/Bib/Lernort_und_Services/Einwilligung_der_Erziehungsberechtigten_fuer_minderjaehrige_Biblitheksbenutzer.pdf"

        return hinweise + info

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


async def bibliothek_OPAC_Online_Public_Access_Catalogue() -> str:
    url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/suchen-und-finden/"

    try:
        soup = await call_browser(url=url)
        side_section = soup.find("main", class_="section-column maincontent-wrap", role="main")

        info = "\nIch bin Benutzer/in der:\nOTH-Bibliothek Amberg: https://opac.oth-aw.de/TouchPoint/start.do?View=faw&Branch=0&Language=de\nOTH-Bibliothek Weiden: https://opac.oth-aw.de/TouchPoint/start.do?View=faw&Branch=2&Language=de"
        hinweise = side_section.find("section", id="opac").get_text() + info
        info = "\nHier ist noch der Link zu einer Schritt für Schritt Anleitung:\nAmberg: https://www.oth-aw.de/files/oth-aw/Einrichtungen/Bib/Suchen_und_Finden/Infoblatt_OPAC_AM.pdf\nWeiden: https://www.oth-aw.de/files/oth-aw/Einrichtungen/Bib/Suchen_und_Finden/Infoblatt_OPAC_WEN.pdf"
        return hinweise + info

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


async def bibliothek_digitale_bibliothek() -> str:
    url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/suchen-und-finden/"

    try:
        soup = await call_browser(url=url)
        side_section = soup.find("main", class_="section-column maincontent-wrap", role="main")

        hinweise = side_section.find("section", id="digitale-bibliothek").get_text()
        info = "\nLink zum VPN-Forticlient: https://www.oth-aw.de/hochschule/services/online-services/wlan-vpn-netzwerk/#c39435\nLink zum E-Book-Infoblatt: https://www.oth-aw.de/files/oth-aw/Einrichtungen/Bib/Suchen_und_Finden/Infoblatt_E-Books.pdf"
        return hinweise + info

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


async def bibliothek_datenbanken() -> str:
    url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/suchen-und-finden/"

    try:
        soup = await call_browser(url=url)
        side_section = soup.find("main", class_="section-column maincontent-wrap", role="main")

        hinweise = side_section.find("section", id="datenbanken").get_text()
        info = "\nAufgrund der Menge an Daten wäre es gut wenn du selber einen Blick auf die Webseite wirfst: https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/suchen-und-finden/#datenbanken"
        return hinweise + info

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


async def bibliothek_zeitschriften() -> str:
    url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/suchen-und-finden/"

    try:
        soup = await call_browser(url=url)
        side_section = soup.find("main", class_="section-column maincontent-wrap", role="main")

        hinweise = side_section.find("section", id="zeitschriften-und-zeitungen").get_text()
        info = "\nInfoblatt Kopienfernleihe (mit Schritt-für-Schritt Anleitung): https://www.oth-aw.de/files/oth-aw/Einrichtungen/Bib/Aktuelles/Infoblatt_Kopienfernleihe.pdf"
        return hinweise + info

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


async def bibliothek_zitiertool_citavi() -> str:
    url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/suchen-und-finden/"

    try:
        soup = await call_browser(url=url)
        side_section = soup.find("main", class_="section-column maincontent-wrap", role="main")

        civitaii = side_section.find("section", id="literaturverwaltung")
        auffuellen_ = civitaii.find("div", id="c428").get_text()
        notes = """Das Wichtigste zu Citavi finden Sie hier:
https://www.citavi.com/de/support/erste-schritte
https://www.oth-aw.de/files/oth-aw/Einrichtungen/Bib/Suchen_und_Finden/Infoblatt_Citavi.pdf
https://www.oth-aw.de/index.php?eID=dumpFile&t=f&f=69313&token=4bcdad08982f9c8c4baf3b1ba4f87edd51e42cfc
https://moodle.oth-aw.de/course/view.php?id=2619
https://youtu.be/9GDYEGWooXE
        """

        return auffuellen_ + "\n" + notes

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


async def bibliothek_zitiertool_alternativen() -> str:
    url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/suchen-und-finden/"

    try:
        soup = await call_browser(url=url)
        side_section = soup.find("main", class_="section-column maincontent-wrap", role="main")

        all = side_section.find("section", id="literaturverwaltung")
        auffuellen_ = all.find("div", id="c19463").get_text()

        notes = """Alternativen zu Citavi finden Sie hier:
https://www.zotero.org/
https://www.mendeley.com/\n
        """

        return notes + auffuellen_ + "Vergleich von different tools: https://mediatum.ub.tum.de/doc/1316333/1316333.pdf"

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


async def bibliothek_ausleihen() -> str:
    url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/ausleihe/"

    try:
        soup = await call_browser(url=url)
        side_section = soup.find("main", class_="section-column maincontent-wrap", role="main")

        all = side_section.find("section", id="leihfristen")
        leihfristen = all.find("div", id="c430").get_text()

        all = side_section.find("section", id="verlaengerung-und-mahnungen")
        verlaengerungen_mahnungen = all.find("div", id="c431").get_text()

        all = side_section.find("section", id="fernleihe")
        fernleihe = all.find("div", id="c432").get_text()
        externe_fernleihe = all.find("div", id="c26718").get_text()
        externe_fernleihe_info = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/lernort-bibliothek-und-services/#hinweise-fuer-externe-benutzer"

        all = side_section.find("section", id="bestand")
        bestand = all.find("div", id="c433").get_text()

        return leihfristen + "\n" + verlaengerungen_mahnungen + "\n" + fernleihe + "\n" + externe_fernleihe + "\n" + externe_fernleihe_info + "\n" + bestand

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


async def bibliothek_FAQ(user_question: str) -> list[tuple[str, str, float]]:
    question_answer_pair = crawl_FAQ(user_question,
                                     url="https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/bibliothek/faq/")
    top_n_similar_questions = find_similar_sentence(question_answer_pair, user_question, 5)
    return top_n_similar_questions


# Rechenzentrum

async def rechenzentrum_news() -> str:
    url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/rechenzentrum/ueber-uns/"

    try:
        soup = await call_browser(url=url)
        side_section = soup.find("main", class_="section-column maincontent-wrap", role="main")

        all = side_section.find("section", id="aktuelles")
        aktuelle_news = all.find("div", id="c34968").get_text()

        info_1 = ("Wichtig: Jeder Student sollte sich den RSS-Feed abonnieren. Dadurch werden automatisch die "
                  "aktuellen nachrichten an die Email weitergeleitet. Studenten müsen sich hier anmelden und dann "
                  "können sie dies verwalten. Link: "
                  "https://www.oth-aw.de/hochschule/services/online-services/schwarzes-brett/abonnement/ ")

        info_2 = ("Hier sind alle aktuellen News und auch die älteren Nachrichten zu finden: "
                  "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/rechenzentrum/news/")

        return info_1 + "\n" + aktuelle_news + "\n" + info_2

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


async def crawl_all_categories_rechenzentrum():
    links = ["https://support.oth-aw.de/help/de-de/8-account-und-login", "https://support.oth-aw.de/help/de-de/9-wlan",
             "https://support.oth-aw.de/help/de-de/10-e-mail",
             "https://support.oth-aw.de/help/de-de/11-laufwerke-dateien-und-ordner",
             "https://support.oth-aw.de/help/de-de/13-telefonie-und-fax",
             "https://support.oth-aw.de/help/de-de/14-sonstiges",
             "https://support.oth-aw.de/help/de-de/16-macos",
             "https://support.oth-aw.de/help/de-de/18-synchronisierung-mobiler-endgerate",
             "https://support.oth-aw.de/help/de-de/19-chatraume", "https://support.oth-aw.de/help/de-de/21-zertifikate",
             "https://support.oth-aw.de/help/de-de/23-office-365", "https://support.oth-aw.de/help/de-de/25-vpn",
             "https://support.oth-aw.de/help/de-de/26-computerlabore"]

    questions_with_links = []

    for url in links:
        soup = await call_browser(url=url)
        question_items = soup.find_all("li", class_="section")

        for item in question_items:
            link = item.find("a")
            if link:
                href = "https://support.oth-aw.de" + link.get("href")
                question = link.get_text(strip=True)
                questions_with_links.append((question, href))

    return questions_with_links


async def rechenzentrum_FAQ(category: str):
    titel_link = crawl_all_categories_rechenzentrum()
    try:
        top_n_similar_questions = find_similar_sentence(titel_link, category, 1)
        url = top_n_similar_questions[0][1]
        soup = await call_browser(url)

        title = soup.find("h1").get_text(strip=True)
        content_div = soup.find("div", class_="article-content")
        article_text = content_div.get_text()
        time_tag = soup.find("div", class_="article-meta").find("time")
        visible_date = time_tag.get_text(strip=True)

        return {"last_updated": visible_date, "content": title + "\n" + article_text,
                "link": top_n_similar_questions[0][1]}
    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


# OTH AW App

async def oth_aw_app() -> dict:
    url = "https://www.oth-aw.de/hochschule/services/online-services/oth-aw-app/"

    try:
        soup = await call_browser(url=url)

        paragraphs = soup.select(".ce-bodytext > p")
        description_text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        list_items = soup.select("ul.list-normal > li")
        features = [li.get_text(strip=True) for li in list_items]

        return {
            "description": description_text,
            "features of the app": features
        }

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return {f"Please check": str(url)}


# schwarzes Brett

async def _extract_brett_items(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    items = []
    for li in soup.select("ul.list-group > li.info"):
        title = li.find("h2").get_text(strip=True)
        date = li.find("time")["datetime"]
        categories = [
            a.get_text(strip=True)
            for a in li.select("ul.list-group-horizontal li a")
        ]
        paragraphs = [
            p.get_text(strip=True)
            for p in li.select(".ce-bodytext p")
            if p.get_text(strip=True)
        ]
        content = "\n".join(paragraphs)

        items.append({
            "title": title,
            "date": date,
            "categories": categories,
            "content": content
        })

    return items


async def oth_schwarzes_brett() -> list[dict]:
    """Nehme nur die aktuelle erste Seite vom schwarzen Brett"""
    url = "https://www.oth-aw.de/hochschule/services/online-services/schwarzes-brett/"

    try:
        soup = await call_browser(url=url, use_cache=False)
        section = soup.find("ul", class_="list-group")
        return _extract_brett_items(str(section))

    except Exception as e:
        logger.error(f"Failed to extract items from URL '{url}': {str(e)}", exc_info=True)
        return []


# Terminplaner der OTH

async def oth_aw_terminplaner() -> str:
    info = """Der offizielle Terminplaner der OTH ist hier zu finden: https://terminplaner.dfn.de/ Man kann hier auch 
    Umfragen erstellen."""
    return info


# Unterstützung für die Lehre

async def _extract_content_and_links(html: Tag) -> dict:
    soup = html

    # Beide mögliche Container: ce-bodytext und textmedia-text
    paragraphs = [
        p.get_text(strip=True)
        for p in soup.select(".ce-bodytext p, .textmedia-text p")
        if p.get_text(strip=True)
    ]
    content = "\n".join(paragraphs)

    links = [
        {
            "href": a["href"] if "https://www.oth-aw.de" in a["href"] else "https://www.oth-aw.de" + a["href"],
            "text": a.get_text(strip=True)
        }
        for a in soup.select(".ce-bodytext a[href], .textmedia-text a[href]")
    ]

    return {
        "content": content,
        "links": links
    }


async def OTH_Support_for_teaching() -> list[dict]:
    url = "https://www.oth-aw.de/hochschule/services/online-services/unterstuetzung-fuer-die-lehre/"

    content = []
    try:
        soup = await call_browser(url=url)

        main = soup.find("div", id="page-content")
        if main is None:
            raise ValueError("Main content container 'page-content' not found.")

        blocks = {
            "kompetenzzentrum_digitale_lehre": "c38468",
            "hochschuldidaktische": "c38466",
            "digitale_aufgaben": "c38469"
        }

        for name, block_id in blocks.items():
            block = main.find("div", id=block_id)
            if block is None:
                logger.warning(f"Block '{name}' with id '{block_id}' not found.")
                continue

            content.append(_extract_content_and_links(block))

        return content

    except Exception as e:
        logger.error(f"Failed to extract items from URL '{url}': {str(e)}", exc_info=True)
        return []


async def get_computerlabore() -> str:
    url = "https://www.oth-aw.de/hochschule/services/online-services/computerlabore/"

    try:
        soup = await call_browser(url=url)
        section = soup.find("div", class_="section section-default")

        kurzanleitung_remotezugriff = section.find("div", id="c38245").get_text()
        verwendung_der_software = section.find("div", id="c38246").get_text()
        verwendung_der_software2 = section.find("div", id="c38247").get_text()
        hinweise2 = section.find("div", id="c39330")
        fernzugriff = hinweise2.find_all("div", class_="accordion-item search-result results-entry")[-1].get_text()
        info = f"Welche Software ist auf den Rechnern in den Laboren in Amberg / Weiden installiert? Siehe: {url}"
        return kurzanleitung_remotezugriff + "\n" + verwendung_der_software + "\n" + verwendung_der_software2 + "\n" + fernzugriff + "\n" + info

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


async def get_videoproduktion() -> str:
    url = "https://www.oth-aw.de/hochschule/services/online-services/videoproduktion/"

    try:
        soup = await call_browser(url=url)
        section = soup.find("div", class_="section section-default")

        tools = section.find("div", id="c16655").get_text(separator=" ")
        vorlesungsaufzeichnung = section.find("div", id="c17584").get_text(separator=" ")
        aufzeichnung_tools = section.find("div", id="c38176").get_text(separator=" ")
        with_video = section.find("div", id="c16754")
        moodle_einbindung = with_video.find("div", class_="ce-bodytext").get_text()
        tips = section.find("div", id="c16708").get_text(separator=" ")

        return tools + "\n" + vorlesungsaufzeichnung + "\n" + aufzeichnung_tools + "\n" + moodle_einbindung + "\n" + tips

    except Exception as e:
        logger.error(f"Failed to extract links from URL '{url}': {str(e)}", exc_info=True)
        return f"Please check: {url}"


##################################
# Vorlesungsfreie Zeit


async def get_wichtige_termine() -> Dict[str, Dict[str, str]]:
    """Extrahiert wichtige Termine wie Vorlesungsfreie Zeiten, Notenbekanntgabe etc. für beide Semester."""

    logger.debug("Starting get_wichtige_termine()...")
    track_function()

    url = "https://www.oth-aw.de/studium/im-studium/organisatorisches/vorlesungs-pruefungs-vorlesungsfreie-zeiten/"
    logger.info(f"Accessing URL: {url}")

    try:
        soup = await call_browser(url)
        logger.debug("Page content retrieved successfully")

        main = soup.find("section", id="mainsection")

        # Alle Überschriften (Semester) und zugehörige Tabellen finden
        result = {}

        # Alle relevanten Container (beide Semesterblöcke)
        blocks = main.find_all('div', class_='frame-inner')

        for block in blocks:
            header = block.find('h2', class_='element-header')
            table = block.find('table', class_='table table-striped')

            if header and table:
                semester_title = header.get_text(strip=True)
                rows = table.find_all('tr')

                term_data = {}
                for row in rows:
                    th = row.find('th')
                    td = row.find('td')
                    if th and td:
                        key = th.get_text(strip=True)
                        value = td.get_text(separator=' ', strip=True)
                        term_data[key] = value

                result[semester_title] = term_data

        return result

    except Exception as e:
        logger.error(f"Error in get_wichtige_termine: {e}")
        return {}


##############################
# Kompetenzzentren

async def get_einrichtung_menu_items(kategorie: str) -> list[dict] | str:
    """Hier gibt es eine liste aus dict zurück mit dem Menünamen und der dahinterliegenden URL.

    Diese URLs sind alle zugelassen. Hier muss das Sprachmodell dann eine kategorie auswählen:

    - "innovations-kompetenzzentrum-kuenstliche-intelligenz",
    - "kompetenzzentrum-digitale-lehre",
    - "kompetenzzentrum-bayern-mittel-osteuropa",
    - "kompetenzzentrum-fuer-gesundheit-im-laendlichen-raum",
    - "kompetenzzentrum-grundlagen-ccg",
    - "kompetenzzentrum-fuer-kraft-waerme-kopplung"

    # url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/kompetenzzentrum-digitaler-campus/" # Sonderfall

    """
    if kategorie == "innovations-kompetenzzentrum-kuenstliche-intelligenz":
        url = "https://www.oth-aw.de/forschung/forschungseinrichtungen/kompetenzzentren/innovations-kompetenzzentrum-kuenstliche-intelligenz/ueber-das-ikki/"
    elif kategorie == "kompetenzzentrum-digitale-lehre":
        url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/kompetenzzentrum-digitale-lehre/"
    elif kategorie == "kompetenzzentrum-bayern-mittel-osteuropa":
        url = "https://www.oth-aw.de/international/internationales-profil/kompetenzzentrum-bayern-mittel-osteuropa/ueber-uns/"
    elif kategorie == "kompetenzzentrum-fuer-gesundheit-im-laendlichen-raum":
        url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/kompetenzzentrum-fuer-gesundheit-im-laendlichen-raum/ueber-das-kompetenzzentrum-fuer-gesundheit-im-laendlichen-raum-kglr/"
    elif kategorie == "kompetenzzentrum-grundlagen-ccg":
        url = "https://www.oth-aw.de/hochschule/ueber-uns/einrichtungen/kompetenzzentrum-grundlagen-ccg/ueber-das-ccg/"
    elif kategorie == "kompetenzzentrum-fuer-kraft-waerme-kopplung":
        url = "https://www.oth-aw.de/forschung/forschungseinrichtungen/kompetenzzentren/kompetenzzentrum-fuer-kraft-waerme-kopplung/aktuelles/"
    else:
        info = "Only one of the following option is allowed: [innovations-kompetenzzentrum-kuenstliche-intelligenz, kompetenzzentrum-digitale-lehre, kompetenzzentrum-bayern-mittel-osteuropa, kompetenzzentrum-fuer-gesundheit-im-laendlichen-raum, kompetenzzentrum-grundlagen-ccg, kompetenzzentrum-fuer-kraft-waerme-kopplung]"
        return info

    try:
        soup = await call_browser(url=url)

        # Menü-Container finden
        menu_container = soup.find("ul", class_="subnav-nav")
        if not menu_container:
            raise ValueError("Menü konnte nicht gefunden werden.")

        menu_items = []
        for li in menu_container.find_all("li", class_="subnav-item"):
            a_tag = li.find("a", class_="subnav-link")
            if a_tag:
                title = a_tag.get("title", "").strip()
                href = a_tag.get("href", "").strip()
                full_url = "https://www.oth-aw.de" + href
                menu_items.append({
                    "title": title,
                    "url": full_url
                })

        return menu_items

    except Exception as e:
        logger.error(f"Fehler beim Extrahieren der Menüeinträge: {e}", exc_info=True)
        return []


async def get_webpage_as_markdown(url: str) -> str:
    source = await call_browser(url)

    side_section = source.find("main", class_="section-column maincontent-wrap", role="main")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = os.path.join(tmpdir, "tempfile.html")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(str(side_section))
        # Use temp_path with Docling here
        conv_res = DocumentConverter().convert(temp_path)
        return conv_res.document.export_to_markdown()


if __name__ == "__main__":
    a = oth_schwarzes_brett()

    # a = get_webpage_as_markdown(a[2]["url"])
    print(a)
