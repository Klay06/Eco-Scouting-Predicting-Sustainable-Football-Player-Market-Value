from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import time

def slow_scroll(driver, pause=0.5, scroll_amount=300, max_scrolls=10):
    """Slow scroll to ensure dynamic content is loaded."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(max_scrolls):
        driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

# Load player names
with open("champions_league_players_2024_2025.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    players = [row[0] for row in reader]

# Setup browser
options = Options()
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920,1080')
# options.add_argument('--headless')  # Uncomment if needed

driver = webdriver.Chrome(options=options)
driver.get("https://www.sofascore.com/")
time.sleep(3)

# Accept cookies
try:
    consent_button = driver.find_element(By.XPATH, '/html/body/div[4]/div[2]/div[2]/div[2]/div[2]/button[1]')
    consent_button.click()
    time.sleep(1)
except Exception:
    pass

# Confirm language
try:
    lang_button = driver.find_element(By.XPATH, '//*[@id="portals"]/div/div/div/div[4]/button')
    lang_button.click()
    time.sleep(1)
except Exception:
    pass

output = []

for player_name in players:
    print(f"\n🔍 {player_name}")
    try:
        driver.get("https://www.sofascore.com/")
        time.sleep(2)

        # Search
        search_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="search-input"]'))
        )
        search_input.clear()
        search_input.send_keys(player_name)
        time.sleep(3)

        # Click first result
        result_xpath = '//*[@id="__next"]/header/div/div[2]/div/div[2]/div/div/div[2]/div/div/div/div/a/div'
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, result_xpath))).click()

        # Wait for page to load
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="__next"]/main')))
        time.sleep(3)

        slow_scroll(driver)

        # Age
        age = "Not found"
        try:
            age_elements = driver.find_elements(By.CLASS_NAME, 'beCNLk')
            for el in age_elements:
                if 'yrs' in el.text or 'anni' in el.text:
                    age = el.text.strip()
                    break
        except:
            pass

        # Team
        try:
            team = driver.find_element(By.XPATH, '//div[@color="onSurface.nLv1"]').text.strip()
        except:
            team = "Not found"

        # ASR (from aria-valuenow)
        try:
            asr_span = driver.find_element(By.XPATH, '/html/body/div[1]/main/div[2]/div/div[2]/div[1]/div[3]/div[2]/div[2]/div/div[2]/div[1]/div[1]/div[2]/div[2]/div/div[1]/div[1]/div[5]/div/div/span')
            asr = asr_span.get_attribute("aria-valuenow")
        except:
            asr = "Not found"

        # Market value
        try:
            market_value = driver.find_element(By.XPATH, '//*[@id="__next"]/main/div[2]/div/div[2]/div[1]/div[2]/div/div[1]/div[3]/div/div[1]/div[2]').text.strip()
        except:
            market_value = "Not found"

        # Nationality
        try:
            nationality = driver.find_element(By.XPATH, '//*[@id="__next"]/main/div[2]/div/div[2]/div[1]/div[2]/div/div[1]/div[2]/div[1]/div[2]/div').text.strip()
        except:
            nationality = "Not found"

        # Transfer count
        try:
            transfer_div = driver.find_element(By.CLASS_NAME, "jaENIm")
            transfer_count = len(transfer_div.find_elements(By.TAG_NAME, "a"))
        except:
            transfer_count = "Not found"

        print(f"✅ {player_name}: Age={age}, Team={team}, ASR={asr}, MV={market_value}, Nationality={nationality}, Transfers={transfer_count}")
        output.append([player_name, age, team, asr, market_value, nationality, transfer_count])

    except Exception as e:
        print(f"❌ Error for {player_name}: {e}")
        output.append([player_name, "Error", "Error", "Error", "Error", "Error", "Error"])

# Save to CSV
with open("players_full_data.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Player Name", "Age", "Team", "ASR", "Market Value", "Nationality", "Transfer History Count"])
    writer.writerows(output)

print(f"\n✅ Done. {len(output)} players saved to CSV.")
driver.quit()
