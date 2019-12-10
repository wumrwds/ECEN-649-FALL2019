import datetime
import glob
import os

import adaboost
from haar_features import FeatureType, gen_features
from integral_image import file_to_integral_img
from util import *

print("\n=========== Start Running ===========\n")

dir_path = os.path.dirname(os.path.realpath(__file__))

print("\n+++++ generate possible features +++++\n")

SIZE = (19, 19)

features_2h = gen_features(SIZE[0], SIZE[1], FeatureType.TWO_HORIZONTAL)
features_2v = gen_features(SIZE[0], SIZE[1], FeatureType.TWO_VERTICAL)
features_3h = gen_features(SIZE[0], SIZE[1], FeatureType.THREE_HORIZONTAL)
features_3v = gen_features(SIZE[0], SIZE[1], FeatureType.THREE_VERTICAL)
features_4 = gen_features(SIZE[0], SIZE[1],  FeatureType.FOUR)
features = features_2h + features_2v + features_3h + features_3v + features_4

print("Number of two-rectangle-horizontal features: %d" % len(features_2h))
print("Number of two-rectangle-vertical features: %d" % len(features_2v))
print("Number of three-rectangle-horizontal features: %d" % len(features_3h))
print("Number of three-rectangle-vertical features: %d" % len(features_3v))
print("Number of four-rectangle features: %d" % len(features_4))
print("Total number of features: %d\n" % len(features))

train_face_images = glob.glob(os.path.join(dir_path + "/dataset/trainset/faces", '**', '*.png'), recursive=True)
train_non_faces_images = glob.glob(os.path.join(dir_path + "/dataset/trainset/non-faces", '**', '*.png'), recursive=True)
test_face_images = glob.glob(os.path.join(dir_path + "/dataset/testset/faces", '**', '*.png'), recursive=True)
test_non_faces_images = glob.glob(os.path.join(dir_path + "/dataset/testset/non-faces", '**', '*.png'), recursive=True)

# train_face_images = glob.glob(os.path.join(dir_path + "/dataset/trainset_copy/faces", '**', '*.png'), recursive=True)
# train_non_faces_images = glob.glob(os.path.join(dir_path + "/dataset/trainset_copy/non-faces", '**', '*.png'), recursive=True)
# test_face_images = glob.glob(os.path.join(dir_path + "/dataset/testset_copy/faces", '**', '*.png'), recursive=True)
# test_non_faces_images = glob.glob(os.path.join(dir_path + "/dataset/testset_copy/non-faces", '**', '*.png'), recursive=True)

train_face_integral_images = file_to_integral_img(train_face_images)
train_non_face_integral_images = file_to_integral_img(train_non_faces_images)
test_face_integral_images = file_to_integral_img(test_face_images)
test_non_face_integral_images = file_to_integral_img(test_non_faces_images)


def features_to_mat(feature, integral_images):
    mat = []
    for img in integral_images:
        feature_row = []
        feature_row.extend([f(img) for f in feature])
        mat.append(feature_row)
        print("Finish converting feature row: ", datetime.datetime.now())

    return mat


train_face_features = features_to_mat(features, train_face_integral_images)
train_face_labels = [1.] * len(train_face_features)

train_non_face_features = features_to_mat(features, train_non_face_integral_images)
train_non_face_labels = [-1.] * len(train_non_face_features)

test_face_features = features_to_mat(features, test_face_integral_images)
test_non_face_features = features_to_mat(features, test_non_face_integral_images)

adaboost_classifiers = adaboost.train(train_face_features + train_non_face_features, train_face_labels + train_non_face_labels, 10)


def output_round(features, ada_classifiers, round):
    best_feature_idx = ada_classifiers[round-1]['index']
    best_feature = features[best_feature_idx]

    print("\nAdaboost rounds: %d" % round)

    # print image with top feature rectangle
    if best_feature.type == FeatureType.TWO_HORIZONTAL:
        printed_img = draw_feature_2h(open_face(test_face_images[0]), best_feature)
        print("\nType: TWO_HORIZONTAL")
    elif best_feature.type == FeatureType.TWO_VERTICAL:
        printed_img = draw_feature_2v(open_face(test_face_images[0]), best_feature)
        print("Type: TWO_VERTICAL")
    elif best_feature.type == FeatureType.THREE_HORIZONTAL:
        printed_img = draw_feature_3h(open_face(test_face_images[0]), best_feature)
        print("Type: THREE_HORIZONTAL")
    elif best_feature.type == FeatureType.THREE_VERTICAL:
        printed_img = draw_feature_2v(open_face(test_face_images[0]), best_feature)
        print("Type: THREE_VERTICAL")
    elif best_feature.type == FeatureType.FOUR:
        printed_img = draw_feature_4(open_face(test_face_images[0]), best_feature)
        print("Type: FOUR")

    print("Position: (%d, %d)" % (best_feature.x, best_feature.y))
    print("Width: %d" % best_feature.width)
    print("Height: %d" % best_feature.height)
    print("Threshold: %f" % ada_classifiers[round-1]['thresh'])
    print("Training accuracy: %f" % ada_classifiers[round-1]['accuracy'])

    printed_img.show()

    # classify
    face_prediction = adaboost.classify(test_face_features, ada_classifiers)
    non_face_prediction = adaboost.classify(test_non_face_features, ada_classifiers)

    false_positive_cnt = count_labels(face_prediction[0], -1.)
    false_negative_cnt = count_labels(non_face_prediction[0], 1.)
    sample_num = len(test_face_features) + len(test_non_face_features)
    test_accuracy = (0. + sample_num - false_positive_cnt - false_negative_cnt) / sample_num

    print("Total accuracy: %f" % test_accuracy)
    print("False Positive: %d" % false_positive_cnt)
    print("False Negative: %d" % false_negative_cnt)


# round 1
output_round(features, adaboost_classifiers[0:1], 1)

# round 3
output_round(features, adaboost_classifiers[0:3], 3)

# round 5
output_round(features, adaboost_classifiers[0:5], 5)

# round 10
output_round(features, adaboost_classifiers[0:10], 10)

# use false-positive rate and false-negative rate to train
print("\n+++++ Train with empirical error +++++")
output_round(features, adaboost_classifiers[0:5], 5)

print("\n+++++ Train with the false-positive rate +++++")
adaboost_classifiers_positive = adaboost.train(train_face_features + train_non_face_features, train_face_labels + train_non_face_labels, 5, 1)
output_round(features, adaboost_classifiers_positive[0:5], 5)

print("\n+++++ Train with the false-negative rate +++++")
adaboost_classifiers_negative = adaboost.train(train_face_features + train_non_face_features, train_face_labels + train_non_face_labels, 5, 2)
output_round(features, adaboost_classifiers_negative[0:5], 5)
