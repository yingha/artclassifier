import numpy as np
import pandas as pd
from tensorflow.keras.applications import mobilenet_v2
from geopy.geocoders import Nominatim
import folium
from folium.plugins import MiniMap

def art_style_classifier(image,model,styles):
    # reverse color channels
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    # reshape image to (1, 224, 224, 3)
    image = image.reshape((1, 224, 224, 3))
    
    # apply pre-processing
    image = mobilenet_v2.preprocess_input(image)

    # make the prediction
    y_pred = model.predict(image)

    pre1 = [f'{(y_pred[0][0]*100).round(2)} %']
    pre2 = [f'{(y_pred[0][1]*100).round(2)} %']
    pre3 = [f'{(y_pred[0][2]*100).round(2)} %']
    pre4 = [f'{(y_pred[0][3]*100).round(2)} %']
    
    artpredictions = pd.DataFrame(list(zip(pre1,pre2,pre3,pre4)),columns=styles)

    return artpredictions

def art_map(art_loc_list):
    '''
    combine multiple rows (artists) into a single row (artist_list) with pandas
    
    '''
    
    # Create a map using the Map() function and the coordinates for Boulder, CO
    loc = Nominatim(user_agent="my_map").geocode('Mali')
    df_loc = pd.DataFrame(loc.raw)
    df_loc.drop([1,2,3], inplace=True)
    lat = float(df_loc['lat'][0])
    lon = float(df_loc['lon'][0])
    m = folium.Map(location=[lat, lon],zoom_start=2.3)

    # mark the museums
    art_loc = art_loc_list.groupby(['museum_name','lat','lon'])['artist'].apply(','.join).reset_index()
    art_loc['link']= art_loc_list.groupby(['museum_name','lat','lon'])['museum_link'].apply(','.join).reset_index()['museum_link']

    for lat,lon,name,link,artist in zip(art_loc['lat'],art_loc['lon'],art_loc['museum_name'],art_loc['link'],art_loc['artist']):
        links = link.split(",")
        artists = artist.split(",")
        artist_links = []
        for link,artist in zip(links,artists):
            artist_link = f'<a href={link}>{artist}</a>'
            artist_links.append(artist_link)
            nl = '\n'          
        folium.Marker(
            location =[lat, lon], # coordinates for the marker (Earth Lab at CU Boulder)
            popup=f'{name}{nl}{[artist_link+nl for artist_link in artist_links]}', # pop-up label for the marker
            #icon=folium.Icon()
        ).add_to(m)
    
    minimap = MiniMap()
    m.add_child(minimap)

    return m



if __name__ == '__main__':
    '''
    model = keras.models.load_model('./models/art_styles_model3.h5')
    print(model)
    image = keras.preprocessing.image.load_img('./test_data/test3_cubism.jpg',target_size=(224,224))
    print(image)
    styles = ['Cubism','Expressionism','Impressionism','PopArt'] # model1, model3
    #styles = ['Expressionism', 'Impressionism', 'Cubism', 'Pop_Art']# model2
    print(art_style_classifier(image,model,styles))
    '''

    #cub = pd.read_csv('./data/cubism_museum_list.csv',index_col=0)
    #exp = pd.read_csv('./data/expressionism_museum_list.csv',index_col=0)
    #imp = pd.read_csv('./data/impressionism_museum_list.csv',index_col=0)
    #pop = pd.read_csv('./data/pop_museum_list.csv',index_col=0)
    print(cub)

    map = art_map(pop) 
    print(map) 
    print('done') 

