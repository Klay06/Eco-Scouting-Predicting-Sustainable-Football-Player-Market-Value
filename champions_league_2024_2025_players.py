from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

def scroll_to_bottom(driver, pause=1.0):
    """Scrolls to bottom slowly to load all lazy-loaded content."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollBy(0, 500);")  # Scroll down in steps
        time.sleep(0.3)  # small pause to let content load
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    print("[INFO] Finished scrolling.")

def accept_cookies(driver):
    """Try to click on cookie popup accept button."""
    try:
        accept_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="cmpwelcomebtnyes"]/a'))
        )
        accept_button.click()
        print("[INFO] Cookie popup accepted.")
        time.sleep(1)  # wait after clicking popup
    except Exception as e:
        print("[INFO] No cookie popup found or could not click accept button.")

def scrape_player_names():
    options = webdriver.ChromeOptions()
    # Comment this out if you want to see the browser in action
    # options.add_argument("--headless")
    options.add_argument("--start-maximized")

    # Ignore SSL errors to reduce handshake error logs
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--ignore-ssl-errors")

    print("[INFO] Starting ChromeDriver...")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    player_names = set()

    try:
        for page in range(1, 18):
            url = f"https://www.worldfootball.net/players_list/champions-league-2024-2025/nach-name/{page}/"
            print(f"\n[INFO] Loading page {page}: {url}")

            try:
                driver.get(url)
            except Exception as e:
                print(f"[ERROR] Failed to load page {page}: {e}")
                continue

            # Handle cookie popup if present
            accept_cookies(driver)

            # Scroll slowly to bottom to load all players
            print("[INFO] Scrolling to bottom to load all players...")
            scroll_to_bottom(driver, pause=1.0)

            # Wait a bit after scrolling
            time.sleep(2)

            try:
                # Find all player links with href starting with "/player_summary/"
                links = driver.find_elements(By.XPATH, '//a[starts-with(@href, "/player_summary/")]')
                print(f"[INFO] Found {len(links)} player link elements on page {page}")
            except Exception as e:
                print(f"[ERROR] Could not find player links on page {page}: {e}")
                continue

            for i, link in enumerate(links, start=1):
                try:
                    name = link.text.strip()
                    if name and name not in player_names:
                        player_names.add(name)
                        print(f"[DEBUG] ({i}) Player found: {name}")
                except Exception as e:
                    print(f"[WARNING] Could not extract text from player link #{i} on page {page}: {e}")

        print(f"\n[INFO] Scraping finished. Total unique players found: {len(player_names)}")

    finally:
        print("[INFO] Closing the browser...")
        driver.quit()

    # Save the results in CSV
    try:
        df = pd.DataFrame(sorted(player_names), columns=["Player Name"])
        output_file = "champions_league_players_2024_2025.csv"
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"[INFO] Data saved to {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save CSV: {e}")

if __name__ == "__main__":
    scrape_player_names()
