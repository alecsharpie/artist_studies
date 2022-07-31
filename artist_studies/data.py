import pandas as pd

def get_ddstudies_data():

    studies_csv = pd.read_csv('https://docs.google.com/spreadsheets/d/14xTqtuV3BuKDNhLotB_d1aFlBGnDJOY0BRXJ8-86GpA/export?format=csv&gid=0', skiprows = 3)
    studies_df = studies_csv.iloc[:, 0:12]

    # clean column names
    studies_df.columns = ['l_name', 'f_name',
       'style_represented', 'complete', 'tags', 'yod', 'user', 'style_or_effect', 'sgl_img_folder',
       'cards_folder', 'batch_id', 'notes']

    studies_df.fillna("", inplace = True)

    return studies_df

def get_ddstudies_artists():

    studies_df = get_ddstudies_data()

    # filter out styles and non visual artists
    artist_df = studies_df[studies_df['complete'].str.contains('x') & ~studies_df['style_or_effect'].str.contains('x')]

    print('Number of artists:', len(artist_df))

    artist_df.loc[:, 'key'] = artist_df.apply(lambda row: f"{row['f_name']} {row['l_name']}", axis = 1).str.replace(' ', '_')

    artist_df.loc[:, 'prompt'] = artist_df.apply(lambda row: f"{row['f_name']} {row['l_name']}", axis = 1).str.strip().str.lower()

    artist_df.loc[:, 'prompt'] = artist_df.prompt.str.strip().str.lower()

    return artist_df
