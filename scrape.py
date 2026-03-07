from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import chromedriver_autoinstaller

from bs4 import BeautifulSoup


def scrape_website(url):

    # Automatically install correct ChromeDriver
    chromedriver_autoinstaller.install()

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(options=options)

    try:

        driver.get(url)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        html = driver.page_source

        return html

    finally:
        driver.quit()


def extract_body_content(html):

    soup = BeautifulSoup(html, "html.parser")

    body = soup.body

    if body:
        return str(body)

    return ""


def clean_body_content(body):

    soup = BeautifulSoup(body, "html.parser")

    # remove unnecessary tags
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.extract()

    text = soup.get_text(separator="\n")

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # remove duplicates
    unique_lines = list(dict.fromkeys(lines))

    cleaned = "\n".join(unique_lines)

    return cleaned