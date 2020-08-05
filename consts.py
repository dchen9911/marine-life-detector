
IMG_SIZE = 120
MIN_SIZE = 90

# CLASS_NAMES = ['Acropora tenella', 'Pachyseris', 'Foliose / plate', 'Encrusting',
#                'Porites', 'Seriatopora hystrix', 'Free-living', 
#                'Sub-massive', 'Acropora', 'Echynophyllia', 'Branching',
#                'Galaxea', 'Favites', 'Leptoseris']
# CLASS_NAMES = ['Acropora tenella', 'Pachyseris', 'foliose', 'Encrusting',
#                'Porites', 'Seriatopora hystrix', 'Free-living', 
#                'Sub-massive', 'Acropora', 'Echynophyllia', 'Branching',
#                'Galaxea', 'Favites', 'Leptoseris']

CLASS_NAMES = ['positive', 'negative']
CLASS_ID = 0
USE_GREY = False
# base_path = './sesoko_data/'
# base_path = './urchin_data/'
base_path = './cuttlefish_data/'
weights_dir = base_path + '/weights/'
N_EPOCHS = 5

svm_weights_path = base_path + 'weights/svm_' + str(CLASS_ID) + '.joblib'

if USE_GREY:
    depth = 1
else:
    depth = 3