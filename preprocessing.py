class preprocessing():

    def __init__(self, news): ## define if news is only text or title as well
        self.news = news

    def __getnews__(self):
        return self.news

    def __setnews__(self, news):
        self.news = news

