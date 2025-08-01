import pandas as pd
import numpy as np
import torch
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from PIL import Image, ImageChops
import torchvision.transforms as transforms
class DataInfo:
    def __init__(self, info_path, data_path, random_state=42):
        self.info_path = info_path
        self.data_path = data_path
        self.random_state = random_state
        self.info_df = pd.DataFrame()
        self.kospi200_info_df = pd.DataFrame()
        self.kospi200_codes = list()
        self.train_codes = list()
        self.val_codes = list()
        self.test_codes = list()
    
    def load_info(self):
        try:
            self.info_df = pd.read_csv(self.info_path)
            self.info_df.loc[:, 'listed_date'] = pd.to_datetime(self.info_df['listed_date'], format='%Y%m%d')
            return self
        except FileNotFoundError:
            print(f"File not found: {self.info_path}")
            return self
        
    def get_kospi200_info(self, listed_date=None):
        if self.info_df.empty:
            print("Info DataFrame is empty. Please load the info first.")
            return self
        kospi200 = self.info_df[self.info_df['kospi200'] != 0]
        if listed_date is not None:
            kospi200 = kospi200[kospi200['listed_date'] < listed_date]
        
        self.kospi200_info_df = kospi200.reset_index(drop=True)
        self.kospi200_codes = self.kospi200_info_df['code'].tolist()
        return self
    
    def train_test_split(self, train_size=0.6, val=True, sector=True):
        if not self.kospi200_codes:
            print("Kospi200 codes are empty. Please load kospi200 info first.")
            return self
        
        if sector:
            self.train_codes = set(self.kospi200_info_df.groupby('kospi200').apply(
                lambda x: x.sample(frac=train_size, random_state=self.random_state)
            )['code'].tolist())
            if val:
                temp_df = self.kospi200_info_df[~self.kospi200_info_df['code'].isin(self.train_codes)]
                self.val_codes = set(temp_df.groupby('kospi200').apply(
                    lambda x: x.sample(frac=0.5, random_state=self.random_state)
                )['code'].tolist())
                self.test_codes = set(temp_df['code'].tolist()) - self.val_codes
            else:
                self.test_codes = set(self.kospi200_codes) - self.train_codes
        else:
            self.train_codes = set(pd.Series(self.kospi200_codes).sample(frac=train_size, random_state=self.random_state).tolist())
            if val:
                temp_df = pd.Series(list(set(self.kospi200_codes) - self.train_codes))
                self.val_codes = set(temp_df.sample(frac=0.5, random_state=self.random_state).tolist())
                self.test_codes = set(temp_df.tolist()) - self.val_codes
            else:
                self.test_codes = set(self.kospi200_codes) - self.train_codes
                
        self.train_codes = list(self.train_codes)
        self.val_codes = list(self.val_codes)
        self.test_codes = list(self.test_codes)
        return self

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, candle_count=12, transform=None):
        self.df = df
        self.candle_count = candle_count
        self.transform = transform
        
    def __len__(self):
        return len(self.df) - self.candle_count + 1 - 2
    
    def _trim(self, img):
        bg = Image.new(img.mode, img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        return img.crop(bbox) if bbox else img
    
    def _make_candle_chart(self, df, start):
        up_color = "#ed3738"
        down_color = "#0d7df3"
        marketcolor = mpf.make_marketcolors(up=up_color, down=down_color, 
                                            edge={'up': up_color, 'down': down_color},
                                            wick={'up': up_color, 'down': down_color},
                                            volume={'up': up_color, 'down': down_color},
                                            vcdopcod=True)
        style = mpf.make_mpf_style(base_mpf_style="default",
                                   marketcolors=marketcolor,
                                   facecolor='white',
                                   rc={'xtick.bottom': False, 'xtick.labelbottom': False,
                                       'ytick.left': False, 'ytick.labelleft': False,})
        fig, _ = mpf.plot(df[start:start+self.candle_count], type='candle', style=style, volume=True, ylabel='', ylabel_lower='', returnfig=True, figsize=(2.24, 2.24))
        canvas = FigureCanvas(fig)
        canvas.draw()
        
        buf = BytesIO()
        # fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        fig.savefig(buf, format='png', pad_inches=0)
        buf.seek(0)
        plt.close(fig)
        
        return self._trim(Image.open(buf).convert('RGB'))

    def _calc_yield(self, df, idx):
        yield_rate = (df.iloc[idx+self.candle_count+1]['close'] - df.iloc[idx+self.candle_count]['close']) / df.iloc[idx+self.candle_count]['close'] * 100
        return yield_rate
        
    def __getitem__(self, idx):
        chart_imgs = list()
        for i in range(3):
            chart_img = self._make_candle_chart(self.df, idx+i)
            if self.transform is not None:
                t = self.transform(chart_img)
            else:
                t = transforms.ToTensor()(chart_img)
            chart_imgs.append(t)
        chart_imgs = torch.stack(chart_imgs)
        yield_rate = self._calc_yield(self.df, idx)
        return chart_imgs, yield_rate
