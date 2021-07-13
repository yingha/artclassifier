import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from geopy.geocoders import Nominatim
import folium
from folium.plugins import MiniMap

imp = pd.read_csv('./data/impressionism_museum_list.csv',index_col=0)
exp = pd.read_csv('./data/expressionism_museum_list.csv',index_col=0)
cub = pd.read_csv('./data/cubism_museum_list.csv',index_col=0)
pop = pd.read_csv('./data/pop_museum_list.csv',index_col=0)

def train_art_model(modelname):

    image_size = (224, 224)
    batch_size = 48

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/training",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/training",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    # preprocess the data
    print("preprocess the data")
    augmented_train_ds = train_ds.map(lambda x, y: (keras.applications.mobilenet_v2.preprocess_input(x), y))
    augmented_val_ds = val_ds.map(lambda x, y: (keras.applications.mobilenet_v2.preprocess_input(x), y))

    # build a model
    print("build a model")
    base_model = keras.applications.mobilenet_v2.MobileNetV2(
                                        weights='imagenet', # use imagnet weights
                                        pooling='avg', # to flatten after covnet layers
                                        include_top=False, # only want the base of the model
                                        input_shape=(224,224,3),
                                        alpha=0.35 # the amount of filters from original network, here we only use 35% 
    )
    base_model.trainable = False
    model=keras.Sequential()
    model.add(base_model)
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

    # train the model
    print("train the model")
    epochs = 50
    callback = keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)

    results = model.fit(
        augmented_train_ds, epochs=epochs, verbose=2, callbacks=[callback], validation_data=augmented_val_ds,
    )
    
    print('the prediction order is:',train_ds.class_names)

    # save the model
    print("save the model as",modelname,"in ./models/")
    model.save('./models/')
    
    print('done')

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

    # train_art_model
    #train_art_model('testmodel')


    # test art_map function
    print(cub)
    map = art_map(pop) 
    print(map) 
    print('done') 



