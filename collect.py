import pyshark
from selenium import webdriver
import threading
import time
import asyncio
import os

websites = [
    "http://gitlab.com",
    "http://buzzfeed.com",
    "http://soundcloud.com",
    "http://google.com",
    "http://facebook.com",
    "http://bing.com",
    "http://yahoo.com",
    "http://duckduckgo.com",
    "http://ask.com",
    "http://twitter.com",
    "http://linkedin.com",
    "http://instagram.com",
    "http://pinterest.com",
    "http://amazon.com",
    "http://ebay.com",
    "http://alibaba.com",
    "http://etsy.com",
    "http://walmart.com",
    "http://cnn.com",
    "http://bbc.com",
    "http://nytimes.com",
    "http://foxnews.com",
    "http://youtube.com",
    "http://vimeo.com",
    "http://dailymotion.com",
    "http://twitch.tv",
    "http://hulu.com",
    "http://netflix.com",
    "http://spotify.com",
    "http://pandora.com",
    "http://reddit.com",
    "http://apple.com",
    "http://microsoft.com",
    "http://adobe.com",
    "http://intel.com",
    "http://ibm.com",
    "http://khanacademy.org",
    "http://coursera.org",
    "http://edx.org",
    "http://udemy.com",
    "http://mit.edu",
    "http://usa.gov",
    "http://nasa.gov",
    "http://whitehouse.gov",
    "http://irs.gov",
    "http://cdc.gov",
    "http://webmd.com",
    "http://mayoclinic.org",
    "http://ea.com",
    "http://who.int",
    "http://clevelandclinic.org",
    "http://espn.com",
    "http://nfl.com",
    "http://nba.com",
    "http://mlb.com",
    "http://fifa.com",
    "http://booking.com",
    "http://expedia.com",
    "http://tripadvisor.com",
    "http://airbnb.com",
    "http://kayak.com",
    "http://paypal.com",
    "http://chase.com",
    "http://bankofamerica.com",
    "http://wellsfargo.com",
    "http://citibank.com",
    "http://medium.com",
    "http://tumblr.com",
    "http://quora.com",
    "http://yelp.com",
    "http://allrecipes.com",
    "http://foodnetwork.com",
    "http://hellofresh.com",
    "http://ubereats.com",
    "http://doordash.com",
    "http://indeed.com",
    "http://glassdoor.com",
    "http://monster.com",
    "http://ziprecruiter.com",
    "http://linkedin.com",
    "http://stackoverflow.com",
    "http://github.com",
    "http://wikipedia.org",
    "http://imdb.com",
    "http://craigslist.org",
    "http://dropbox.com",
    "http://weebly.com",
    "http://zoho.com",
    "http://slack.com",
    "http://canva.com",
    "http://chegg.com",
    "http://shopify.com",
    "http://wordpress.com",
    "http://tiktok.com",
    "http://snapchat.com",
    "http://notion.so",
    "http://bitly.com",
    "http://roblox.com",
    "http://epicgames.com",
    "http://steamcommunity.com",
    "http://weather.com",
]

interface = "en0"
duration = 2
folders = ["training", "testing"]
capturesPerSite = [4, 1]

# Ensure the folders exist
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

def capture_packets(interface, output_file):
    capture = pyshark.LiveCapture(interface=interface, output_file=output_file)
    for packet in capture.sniff_continuously():
        if stop_sniffing.is_set():
            break
    print(f"Packet capture saved to {output_file}\n")

stop_sniffing = threading.Event()
for url in websites:
    # start where we left off
    folder = folders[1]
    prefix = url.replace("http://", "").replace(".com", "")
    exists = False
    for file_name in os.listdir(folder):
        if file_name.startswith(prefix):
            exists = True

    if not exists:
        for i, numcapture in enumerate(capturesPerSite):
            for j in range(1, numcapture+1):
                folder = folders[i]
                savefilename = os.path.join(folder, f"{url.replace("http://", "").replace(".com", "")}-{folder}-{j}.pcap")
                print(f"Starting packet capture for {url}, saving to {savefilename}")
                stop_sniffing.clear()

                capture_thread = threading.Thread(target=capture_packets, args=(interface, savefilename))
                capture_thread.start()

                driver = webdriver.Firefox()
                driver.get(url)
                time.sleep(duration)
                stop_sniffing.set()
                driver.quit()

                capture_thread.join(timeout=duration + 1)