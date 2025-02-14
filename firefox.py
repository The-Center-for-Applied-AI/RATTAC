# ESSENTIAL LIBRARIES FOR FIREFOX SETUP
import os
import requests
import tarfile


# FIREFOX DOWNLOAD FOR SELENIUM USE
def download_firefox():
    download_url = f"https://download.mozilla.org/?product=firefox-latest-ssl&os=linux64&lang=en-US"

    print(f"Downloading the latest Firefox.")

    # Download the latest version of Firefox
    tarball_path = "firefox.tar.xz"
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(tarball_path, "wb") as file:
            for chunk in r.iter_content(chunk_size=8192):
                file.write(chunk)

    # Extract the Firefox tarball
    with tarfile.open(tarball_path, "r:xz") as tar:
        tar.extractall()

    # Make Firefox runnable if it exists and get its location
    if 'firefox.tar.xz' in os.listdir():
        os.remove(tarball_path)
        binary_loc = './firefox/firefox'
        os.chmod(binary_loc, 0o755)  # Make it executable
        print("Installation successful.")
    else:
        print("Installation failed.")


# GECKO DRIVER DOWNLOAD FOR SELENIUM USE
def download_gecko():
    # Fetch the latest release version number from GitHub API
    response = requests.get("https://api.github.com/repos/mozilla/geckodriver/releases/latest")
    response.raise_for_status()  # Raise an error for failed requests

    latest_version = response.json()["tag_name"].strip("v")  # Extract version number
    download_url = f"https://github.com/mozilla/geckodriver/releases/download/v{latest_version}/geckodriver-v{latest_version}-linux64.tar.gz"

    print(f"Downloading the latest Geckodriver.")

    # Download the latest version of Geckodriver
    tarball_path = "geckodriver.tar.gz"
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(tarball_path, "wb") as file:
            for chunk in r.iter_content(chunk_size=8192):
                file.write(chunk)

    # Extract Geckodriver the tarball
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall()

    # Make Geckodriver runnable if it exists and get its location
    if "geckodriver.tar.gz" in os.listdir():
        os.remove(tarball_path)
        driver_loc = './geckodriver'
        os.chmod(driver_loc, 0o755)  # Make it executable
        print("Installation successful.")
    else:
        print("Installation failed.")


# RUN THE FOLLOWING IF YOU WISH TO USE UBLOCK ORIGIN WITH FIREFOX'S SELENIUM RUN
def download_ublock():
    # Fetch the latest release version number from GitHub API
    response = requests.get("https://api.github.com/repos/gorhill/uBlock/releases/latest")
    # Raise an error for failed requests
    response.raise_for_status()

    # Extract version number
    latest_version = response.json()["tag_name"]
    download_url = f"https://github.com/gorhill/uBlock/releases/download/{latest_version}/uBlock0_{latest_version}.firefox.signed.xpi"

    print(f"Downloading uBlock Origin")

    # Download the latest version of uBlock Origin
    xpi_path = "ublock.firefox.signed.xpi"
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(xpi_path, "wb") as file:
            for chunk in r.iter_content(chunk_size=8192):
                file.write(chunk)

    if "ublock.firefox.signed.xpi" in os.listdir():
        print("Installation successful.")
    else:
        print("Installation failed.")