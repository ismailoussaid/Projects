
def pred_to_label(prediction, attribute):
    if attribute == 'mustache':
        if prediction == 0:
            return 'no mustache'
        else:
            return 'mustache'

    elif attribute == 'eyeglasses':
        if prediction == 0:
            return 'no eyeglasses'
        else:
            return 'eyeglasses'

    elif attribute == 'beard':
        if prediction == 0:
            return 'no beard'
        else:
            return 'beard'

    elif attribute == 'hat':
        if prediction == 0:
            return 'no hat'
        else:
            return 'wearing hat'

    else:
        if prediction == 0:
            return 'hairy'
        else:
            return 'bald'


def predict(model, test_images, flag='class'):
    predictions, adapted_images = [], []
    predicted_mustache, predicted_eyeglasses, \
    predicted_beard, predicted_hat, predicted_bald = [], [], [], [], []

    for image in test_images:
        img = image.reshape(-1, 36, 36, 1)
        prediction = model.predict(img)
        mustache_predict, eyeglasses_predict, \
        beard_predict, hat_predict, bald_predict = np.argmax(prediction, axis=2)
        if flag == 'class':
            multiple_append([predicted_mustache, predicted_eyeglasses,
                             predicted_beard, predicted_hat, predicted_bald],
                            [mustache_predict[0], eyeglasses_predict[0],
                             beard_predict[0], hat_predict[0], bald_predict[0]])
        elif flag == 'label':
            multiple_append([predicted_mustache, predicted_eyeglasses,
                             predicted_beard, predicted_hat, predicted_bald],
                            [pred_to_label(mustache_predict[0], 'mustache'),
                             pred_to_label(eyeglasses_predict[0], 'eyeglasses'),
                             pred_to_label(beard_predict[0], 'beard'),
                             pred_to_label(hat_predict[0], 'hat'),
                             pred_to_label(bald_predict[0], 'bald')])

    return predicted_mustache, predicted_eyeglasses, \
           predicted_beard, predicted_hat, predicted_bald


def initialize_results(att_dict, compute_flops=True):
    dict_col = {}
    dict_col["number of Conv2D"] = []
    dict_col["number of Dense"] = []
    dict_col["kernel size"] = []
    dict_col["first conv"] = []
    dict_col["second conv"] = []
    dict_col["unit"] = []
    dict_col["batch_size"] = []
    for key, value in att_dict.items():
        dict_col[key + " cv score"] = []
        dict_col[key + " std score"] = []
    if compute_flops == True:
        dict_col["flop"] = []
    return dict_col

