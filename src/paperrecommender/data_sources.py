import requests
import datetime
import feedparser

class DataSource:
    """
    Base class for data sources.
    """

    def __init__(self, period=25, stale_period=0.1):
        """
        Initializes the DataSource with a period in hours.
        """
        self.period = period
        self.stale_period = stale_period
        self.titles, self.abstracts, self.links = self.get_data()
        self.last_refresh = self.current_datetime()

    @staticmethod
    def current_datetime():
        """
        Returns the current UTC datetime.
        """
        return datetime.datetime.now(datetime.timezone.utc)

    def get_data(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def refresh_data(self):
        """
        Refreshes the data by calling get_data method.
        """
        current_date = self.current_datetime()
        # Only refresh if the data is stale
        if self.last_refresh and (current_date - self.last_refresh).total_seconds() < self.stale_period * 3600:
            return
        self.titles, self.abstracts, self.links = self.get_data()   
        self.last_refresh = current_date
    

class ArXivDataSource(DataSource):
    """
    Class to fetch data from ArXiv.
    """
    base_url = "http://export.arxiv.org/api/query?"

    def get_data(self):
        current_date = self.current_datetime()
        start_date_str = (current_date - datetime.timedelta(hours=self.period)).strftime("%Y%m%d%H%M")
        end_date_str = current_date.strftime("%Y%m%d%H%M")
        query_params = {
            "search_query": f"submittedDate:[{start_date_str}+TO+{end_date_str}]",
            "max_results": 2000,
            "sortBy": "lastUpdatedDate",
            "sortOrder": "descending"
        }
        query_url = self.base_url + "&".join([f"{key}={value}" for key, value in query_params.items()])
        print(f'Sending query: {query_url}')
        req = requests.get(query_url)
        parsed_data = feedparser.parse(req.text)
        titles = []
        abstracts = []
        links = []
        for entry in parsed_data.entries:
            titles.append(entry.title)
            abstracts.append(entry.summary)
            links.append(entry.link)
        return titles, abstracts, links
        

        