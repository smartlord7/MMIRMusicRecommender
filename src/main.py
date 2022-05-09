import os
import warnings
from mmir_pipeline.process import *
from features.librosa_wrap.temporal import *
from testing.features import *
from mmir_pipeline.similarity_analysis import gen_distances, rank_query_results


def process_data(process_callback,
                 dir_path=IN_DIR_PATH_ALL_DATABASE,
                 out_path=OUT_PATH_ALL_FEATURES,
                 in_extension=EXTENSION_MP3):
    """
    Function used to process the given data.
    """

    if isfile(out_path):
        return

    data_files = os.listdir(dir_path)
    data_files.sort()
    all_processed = np.empty((len(data_files), N_COLS))
    i = int()
    for data_file_name in data_files:
        if data_file_name.endswith(in_extension):
            print("Processing %s..." % data_file_name)
            data = librosa.load(dir_path + data_file_name, sr=SAMPLING_RATE, mono=IS_AUDIO_MODE_MONO)[0]
            processed = process_callback(data)
            all_processed[i] = processed
            i += 1

    all_processed = normalize_min_max(all_processed)

    np.savetxt(out_path, all_processed, fmt='%f', delimiter=FEATURE_DELIM)


def objective_ranking(querie, n=20):
    with open(PATH_METADATA) as f:
        metadata = f.readlines()
        size = len(metadata)
        metadata = [x.split(",") for x in metadata]
        l = []
        querie = querie.strip(".mp3")

        for i in range(1, size):
            model = metadata[i]

            if model[0].strip("\"'") == querie:
                print("%s: %s" % (model[0], model[1]))

                for j in range(1, size):
                    count = 0
                    current = metadata[j]

                    if current[0].strip("\"'") != querie:

                        # ARTIST
                        if current[1].strip("\"'") == model[1].strip("\"'"):
                            count += 1

                        # GENRE
                        genre = current[11].strip("\"'").split("; ")
                        genre_2 = model[11].strip("\"'").split("; ")

                        genre = set(list(map(lambda x: x.lower(), genre)))
                        genre_2 = set(list(map(lambda x: x.lower(), genre_2)))

                        count += len(genre.intersection(genre_2))

                        # QUADRANT
                        if current[3].strip("\n'") == model[3].strip("\n'"):
                            count += 1

                        # EMOTION
                        emotion = set(current[9].strip("\"'").split("; "))

                        emotion_2 = set(model[9].strip("\"'").split("; "))

                        count += len(emotion.intersection(emotion_2))

                    else:
                        count = -1

                    l.append(count)

        top_index = np.argsort(np.array(l))[len(l):len(l) - n:-1]

        print("Top %d Recommendations: " % n)

        counter = 1

        for i in top_index:
            print("%d - %s: %s by %s" % (counter, metadata[i + 1][0], metadata[i + 1][2], metadata[i + 1][1]))
            counter += 1


def main():
    """
        Main function.
        """
    files = os.listdir(PATH_QUERIES)

    count = 0
    # Objective Ranking
    for querie in files:
        objective_ranking(querie)

    warnings.filterwarnings("ignore")
    default_features = process_default_features(IN_PATH_DEFAULT_FEATURES, OUT_PATH_DEFAULT_FEATURES)
    process_data(featurize)

    for dist in TYPES_DISTANCES:
        gen_distances(dist)
        gen_distances(dist, IN_PATH_DEFAULT_FEATURES, OUT_PATH_DEFAULT_DISTANCES, default_features)

    for query_path in os.listdir(PATH_QUERIES):
        for dist in TYPES_DISTANCES:
            results, dist = rank_query_results(query_path, OUT_PATH_DISTANCES + dist + EXTENSION_CSV, IN_DIR_PATH_ALL_DATABASE)
            for i in range(len(results)):
                print("%d - %s (%.4f)" % (i + 1, results[i], dist[i]))


if __name__ == '__main__':
    main()
