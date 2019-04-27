#
#
#   Public URL mixin
#
#

from ...services.file_downloader import FileDownloader


class PublicURLMixin:


    public_urls = None


    def download(self, download_path):
        if self.public_urls is None:
            raise ValueError("public_urls cannot be None")

        downloader = FileDownloader()
        downloader.download(self.public_urls, download_path, verbose=True)
