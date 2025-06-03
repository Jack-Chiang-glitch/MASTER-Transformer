from pathlib import Path
import pandas as pd
from datetime import date, datetime, timedelta
import os

class FileLoader:
    _cache = {}

    # 副檔名與對應的 pd.read 方法
    _readers = {
        '.csv': pd.read_csv,
        '.ftr': pd.read_feather,
        '.pkl': pd.read_pickle,
        '.pickle': pd.read_pickle,
        '.xlsx': pd.read_excel,
        '.parquet': pd.read_parquet,
        '.json': pd.read_json,
    }

    @staticmethod
    def load(file_path):
        path = Path(file_path)
        ext = path.suffix.lower()

        if path in FileLoader._cache:
            print(f"⚡ 快取使用: {path}")
        else:
            print(f"📂 讀取: {path}")
            reader = FileLoader._readers.get(ext)

            if not reader:
                raise ValueError(f"❌ 不支援的副檔名: {ext}")

            file = reader(path)
            FileLoader._cache[path] = file

        return FileLoader._cache[path]


class StrTool:
    """
    把日期字串轉成像是datetime.date(2015, 6, 1)這種格式
    """
    @staticmethod 
    def to_date(date_str):
        return datetime.strptime(date_str, '%Y-%m-%d').date()
        
    """
    載入ftr的DataFrame(單一個股或是ETF)，並且將日期index轉成datetime格式
    """
    @staticmethod
    def load_ftr_with_date_index(path):
        df = pd.read_feather(path)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.set_index('trade_date')
        return df
    



    
class TradingCalendar:
    """
    給定時間範圍，回傳所有在時間範圍內的交易日
    如果to_string = True，回傳純字串
    如果to_string = False，回傳datetime.date(2025, 2, 3)這種格式
    """
    @staticmethod 
    def trading_days(start_date, end_date=None, to_string=False):
        start_date = StrTool.to_date(start_date)
        end_date = date.max if end_date is None \
                   else StrTool.to_date(end_date)
        
        
        folder_path = r'Z:\kbars_feather\processed\5min_adjusted_1335'
        # 獲得日期字串 eg. ['2024=02-01', '2024-02-03'.......]
        tradedate_list = [f.removesuffix('.ftr') 
                          for f in os.listdir(folder_path)
                          if f.endswith('.ftr')]
        # 單純日期字串轉成datetime 方便後續比較 eg. [datetime.date(2025, 2, 3), datetime.date(2025, 2, 4).....]
        tradedate_list = [StrTool.to_date(date)
                          for date in tradedate_list]
        # 篩選要的日期範圍
        tradedate_list = [d for d in tradedate_list
                          if start_date<= d <=end_date]
        # 排序日期
        tradedate_list.sort()

        if(to_string):
            tradedate_list = [date.strftime('%Y-%m-%d') for date in tradedate_list]
        
        return tradedate_list


class StockUniverse:
    """
    回傳所有上市上櫃公司的代號
    """
    market_type_df = pd.read_pickle(r'Y:\因子回測_江建彰\上市上櫃歷年列表.pkl')
    all_tickers = market_type_df.columns
    Type_list = market_type_df.loc['2025-04-16'].values
    @staticmethod
    def TWSE():
        TWSE_stock_list = [ticker 
                           for ticker, Type in zip(StockUniverse.all_tickers, StockUniverse.Type_list)
                           if Type=='TWSE']
        return TWSE_stock_list
        
    @staticmethod
    def OTC():
        OTC_stock_list = [ticker 
                          for ticker, Type in zip(StockUniverse.all_tickers, StockUniverse.Type_list)
                          if Type=='OTC']
        return OTC_stock_list

    @staticmethod
    def all():
        return StockUniverse.TWSE()+StockUniverse.OTC()


class MarketInfo:
    @staticmethod
    def process(df):
        window_sizes = [5, 10, 20, 30, 60]
        for w in window_sizes:
            df[f'price_mean_{w}'] = df['price'].rolling(window=w).mean()
            df[f'price_std_{w}']  = df['price'].rolling(window=w).std()
        
            df[f'amount_mean_{w}'] = df['adj_amount'].rolling(window=w).mean()
            df[f'amount_std_{w}']  = df['adj_amount'].rolling(window=w).std()
            
        df.dropna(axis=0, inplace=True)
        df.drop(columns=['amount','adj_amount'], inplace=True)
        return df

    
        
    @staticmethod
    def TPEX():
        TPEX_df = pd.read_feather(r'Y:\因子回測_江建彰\櫃買報酬指數及成交金額.ftr')
        TPEX_df['adj_amount'] = TPEX_df['amount']/100000000
        return MarketInfo.process(TPEX_df)
        
    @staticmethod
    def ETF00733():
        ETF00733_df = StrTool.load_ftr_with_date_index(r'Z:\TWSE\adjusted_price\00733.ftr')
        ETF00733_df = ETF00733_df[['adj_close_price', 'amount']]
        ETF00733_df['adj_amount'] = ETF00733_df['amount']/1000000
        ETF00733_df.columns = ['price', 'amount', 'adj_amount']
        return MarketInfo.process(ETF00733_df)

    @staticmethod
    def standardize_fillna_zero(row):
        valid = row[~row.isna()]
        if valid.empty:
            return row.fillna(0)
        standardized = (valid - valid.mean()) / valid.std(ddof=0)
        row.update(standardized)
        return row.fillna(0)

    @staticmethod
    def TPEX_norm():
        TPEX_df = MarketInfo.TPEX()
        # 假設你的 DataFrame 是 df
        cols_to_standardize1 = ['price', 'price_mean_5', 'price_mean_10', 'price_mean_20', 'price_mean_30', 'price_mean_60']
        cols_to_standardize2 = ['price_std_5', 'price_std_10', 'price_std_20', 'price_std_30', 'price_std_60']
        cols_to_standardize3 = ['amount_mean_5', 'amount_mean_10', 'amount_mean_20', 'amount_mean_30', 'amount_mean_60']
        cols_to_standardize4 = ['amount_std_5', 'amount_std_10', 'amount_std_20', 'amount_std_30', 'amount_std_60']
        
        TPEX_df[cols_to_standardize1] = TPEX_df[cols_to_standardize1].apply(MarketInfo.standardize_fillna_zero, axis=1)
        TPEX_df[cols_to_standardize2] = TPEX_df[cols_to_standardize2].apply(MarketInfo.standardize_fillna_zero, axis=1)
        TPEX_df[cols_to_standardize3] = TPEX_df[cols_to_standardize3].apply(MarketInfo.standardize_fillna_zero, axis=1)
        TPEX_df[cols_to_standardize4] = TPEX_df[cols_to_standardize4].apply(MarketInfo.standardize_fillna_zero, axis=1)

        return TPEX_df



class FactorLibrary:
    multi_df = FileLoader.load(r'Y:\因子回測_江建彰\因子庫.pkl')
    multi_df.columns.names = ['factor', 'ticker']
        
    @staticmethod
    def get_factors_for_ticker(ticker, start_date, end_date):
        return FactorLibrary.multi_df.xs(ticker, axis=1, level='ticker').loc[start_date : end_date]

    @staticmethod
    def get_factor_across_tickers(idx, start_date, end_date):
        factor_name = f'factor_{idx}'
        return FactorLibrary.multi_df.loc[start_date : end_date, factor_name]


class FactorLibrary2:
    
    def __init__(self, path):
        self.multi_df = FileLoader.load(path)
        self.multi_df.columns.names = ['factor', 'ticker']
        
        
        
    
    def get_factors_for_ticker(self, ticker, start_date, end_date):
        return self.multi_df.xs(ticker, axis=1, level='ticker').loc[start_date : end_date]

    # useful
    def get_factor_across_tickers(self, idx, start_date, end_date):
        factor_name = f'factor_{idx}'
        return self.multi_df.loc[start_date : end_date, factor_name]
        