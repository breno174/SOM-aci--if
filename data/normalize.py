import pandas as pd
import csv

class Normalize:
    def __init__(self, path):
        self.path = path
        self.data = None

    def load_with_pandas(self, columns_to_keep):
        self.df = pd.read_csv(self.path)
        # Filtra as colunas
        self.df = self.df[columns_to_keep]
        # Converte para array numpy para usar na SOM
        self.data = self.df.values
        return self.data
    
    def usage_coluns(self):
        self.load_with_pandas(["daily_social_media_hours", "sleep_hours", "platform_usage","academic_performance", "physical_activity", "stress_level"])
        ## Trocando coluna de redes sociais por um valor DENTRO do Pandas:
        self.df["platform_usage"] = self.df["platform_usage"].replace({"Instagram": 0, "TikTok": 1, "Both": 2})
        
        # Converte novamente para numpy array para a SOM agora que é numérico
        self.data = self.df.values
        return self.data
    
    def show(self):
        print(self.data)