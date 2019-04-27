#
#
#   Public URL mixin
#
#

from ...services.file_downloader import FileDownloader


class PublicURLMixin:
    """
    Mixin to download public urls. It provides implementation for ``download`` method.

    Attributes
    ----------
    public_urls: str or list of str
        Urls to download.
    """

    public_urls = None


    def download(self, download_path):
        """
        Download urls specified by ``public_urls``.

        Parameters
        ----------
        download_path: str
            Path to download urls.
        """

        if self.public_urls is None:
            raise ValueError("public_urls cannot be None")

        downloader = FileDownloader()
        downloader.download(self.public_urls, download_path, verbose=True)
