import requests
import json
import os
import sys
import math
import secrets

DOWNLOAD_DIRECTORY = "person_images" #/tmp/images
BASEURL = 'https://api.imagemonkey.io/' #'http://127.0.0.1:8081/'
#SEARCH_QUERY = "left/man|right/man|left/woman|right/woman"
SEARCH_QUERY = "man|woman|man/sitting|man/walking|man/standing|man/running|woman/sitting|woman/walking|woman/reclining"

if not hasattr(secrets, 'X_API_TOKEN') or secrets.X_API_TOKEN == "":
	print("Please provide a valid API Token in secrets.py")
	sys.exit(1)

class ImageMonkeyGeneralError(Exception):
    """Base class for exceptions raised by ImageMonkey."""

class Image(object):
    def __init__(self, uuid, width, height):
        self._uuid = uuid
        self._width = width
        self._height = height
        self._path = None
        self._folder = None

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def folder(self):
        return self._folder

    @folder.setter
    def folder(self, folder):
        self._folder = folder

    @property
    def uuid(self):
        return self._uuid

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class ImageMonkeyApi(object):
    def __init__(self, api_version=1, base_url=BASEURL):
        self._api_version = api_version
        self._base_url = base_url

    def export(self, query):
        url = self._base_url + "v" + str(self._api_version) + "/export"
        params = {"query": query}

        r = requests.get(url, params=params, headers={"X-Api-Token": secrets.X_API_TOKEN})
        if(r.status_code == 500):
            raise InternalImageMonkeyAPIError("Could not perform operation, please try again later")
        elif(r.status_code != 200):
            data = r.json()
            raise ImageMonkeyAPIError(data["error"])

        data = r.json()
        res = []
        for elem in data:
            image = Image(elem["uuid"], elem["width"], elem["height"])
            res.append(image)
        return res

    def download_image(self, uuid, folder, extension=".jpg",overwrite=True):
        if not os.path.isdir(folder):
            raise ImageMonkeyGeneralError("folder %s doesn't exist" %(folder,))

        filename = folder + os.path.sep + uuid + extension
        if not overwrite and os.path.exists(filename):
            raise ImageMonkeyGeneralError("image %s already exists in folder %s" %(uuid,folder))

        url = self._base_url + "v" + str(self._api_version) + "/donation/" + uuid
        response = requests.get(url, headers={"X-Api-Token": secrets.X_API_TOKEN})
        if response.status_code == 200:
            print(f"Downloading image {uuid}")
            with open(filename, 'wb') as f:
                f.write(response.content)
        else:
            raise ImageMonkeyAPIError("couldn't download image %s" %(uuid,))

    def get_image_labels(self, image_uuid):
        url = self._base_url + "v" + str(self._api_version) + "/donation/" + image_uuid + "/labels"
        response = requests.get(url, headers={"X-Api-Token": secrets.X_API_TOKEN})
        if response.status_code != 200:
            raise ImageMonkeyGeneralError("couldn't get labels for image with uuid " + image_uuid)
        labels = []
        for entry in response.json():
            labels.append(entry["label"])
            if "sublabels" in entry and entry["sublabels"] is not None:
                for sublabel in entry["sublabels"]:
                    labels.append(sublabel["name"] + "/" + entry["label"])
        return labels

    def get_image_annotations(self, image_uuid):
        url = self._base_url + "v" + str(self._api_version) + "/donation/" + image_uuid + "/annotations"
        response = requests.get(url, headers={"X-Api-Token": secrets.X_API_TOKEN})
        if response.status_code != 200:
            raise ImageMonkeyGeneralError("couldn't get labels for image with uuid " + image_uuid)
        data = response.json()
        return data

if __name__ == "__main__":
    if not os.path.isdir(DOWNLOAD_DIRECTORY):
        print(f"download directory {DOWNLOAD_DIRECTORY} doesn't exist!")
        sys.exit(1)

    if SEARCH_QUERY == "":
        print("Please provide a search query!")
        sys.exit(1)

    imagemonkey_api = ImageMonkeyApi(base_url=BASEURL)
    images = imagemonkey_api.export(SEARCH_QUERY)
    for image in images:
        print(f"downloading image (uuid: {image.uuid}, width: {image.width}, height: {image.height}) to {DOWNLOAD_DIRECTORY}")
        imagemonkey_api.download_image(image.uuid, DOWNLOAD_DIRECTORY,overwrite=True)
        labels = imagemonkey_api.get_image_labels(image.uuid)
        labels_lst = ",".join(labels)
        print(f"image {image.uuid} has the following labels: {labels_lst}")
        annotations = imagemonkey_api.get_image_annotations(image.uuid)
        print(f"image {image.uuid} has the following annotations: {annotations}")
        file = open(DOWNLOAD_DIRECTORY+"/"+str(image.uuid)+".json","w")
        json.dump(annotations,file)
        file.close()