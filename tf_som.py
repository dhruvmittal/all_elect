import pandas as pd
import numpy as np
import tensorflow as tf
import dill


# A self organizing map is a wonderful tool to reduce dimension
# Here, we use it to move from the 4d featurespace of Performance parameters
# to a single number that contains the information of "relative performance"
# When I use this win a little fuzzy bucketing, it really captures the
# essence of pollster performance.
class TF_SOM(object):
    _trained = False
    def __init__(self, m,n,dim,n_iteratations=100,alpha=.3,sigma=None):
        # m by n dimensions of map
        # n_iter for training
        # dim is dimensionality of training input -> 4
        # alpha is inital learning rate
        # sigma is inital neighborhood value
        self._m = m
        self._n = n
        self._n_iterations = n_iteratations
        if sigma is None:
            sigma = max(m,n) / 2.0
        
        self._graph = tf.Graph()
        
        with self._graph.as_default():
            # randomly initialize weightage vectors x each neuron
            self._weightage_vects = tf.Variable(tf.random_normal([m*n,dim]))
            
            #grid location of neurons
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m,n))))
            
            #training vector placeholder
            self._vect_input = tf.placeholder('float',[dim])
            self._iter_input = tf.placeholder('float')
            
            #best matching unit -> calculate euclidean distance bectween
            #each weightage vector and input, return index neuron w/ lowest
            bmu_index = tf.argmin(
                    tf.sqrt(
                        tf.reduce_sum(
                            tf.pow(
                                tf.sub(self._weightage_vects,
                                      tf.pack([self._vect_input
                                               for i in range(m*n)]
                                    )),
                                2
                            ),
                            1
                        )
                    ),
                    0
                )
            
            #extract the location of bmu based on bmu index
            slice_input = tf.pad(
                    tf.reshape(bmu_index, [1]),
                    np.array([[0,1]])
                )
            bmu_loc = tf.reshape(
                    tf.slice(self._location_vects, slice_input,
                            tf.constant(np.array([1,2]))),
                [2])
            
            # adjust alpha, sigma based on iter
            learning_rate_op = tf.sub(1.0, tf.div(
                    self._iter_input, self._n_iterations
                ))
            _alpha_op = tf.mul(alpha,learning_rate_op)
            _sigma_op = tf.mul(sigma, learning_rate_op)
            
            # operation that generates vectors w/ learning rates
            # for all neurons
            bmu_distance_squares = tf.reduce_sum(
                tf.pow(
                    tf.sub(self._location_vects, tf.pack(
                            [bmu_loc for i in range(m*n)]
                        )
                    ),
                    2
                ),
                1
            )
            neighborhood_func = tf.exp(
                tf.neg(
                    tf.div(
                        tf.cast( bmu_distance_squares, "float32"),
                        tf.pow(_sigma_op,2)
                    )
                )
            )
            learning_rate_op = tf.mul(_alpha_op, neighborhood_func)
            
            # operation to update the weightage vectors based on input
            learning_rate_multiplier = tf.pack([tf.tile(
                        tf.slice(learning_rate_op, np.array([i]),
                                 np.array([1])
                                ),
                        [dim]
                    )
                           for i in range(m*n)])
            weightage_delta = tf.mul(
                learning_rate_multiplier,
                tf.sub(
                    tf.pack([self._vect_input for i in range(m*n)]),
                       self._weightage_vects
                )
            )
            new_weightages_op = tf.add(
                self._weightage_vects,
                weightage_delta
            )
            self._training_op = tf.assign(
                self._weightage_vects,
                new_weightages_op
            )
            
            # init session, vars
            self._sess = tf.Session()
            init_op = tf.initialize_all_variables()
            self._sess.run(init_op)
    def _neuron_locations(self,m,n):
        # yields one by one the 2d locations of individual neurons in
        # SOM
        for i in range(m):
            for j in range(n):
                yield np.array([i,j])
                
    def train(self,input_vects):
        # input_vects -> iterable of 1d numpy arrays w/ dimension dim
        
        #training iterations
        for iter_no in range(self._n_iterations):
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                              feed_dict={self._vect_input: input_vect,
                                         self._iter_input: iter_no
                                        }
                  )
        # store centroid grid for retrieval later
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid
         
        # let the model know that it's trained
        self._trained = True
        
    def get_centroids(self):
        if not self._trained:
            raise ValueError("Not trained, no centroids")
        return self._centroid_grid
    
    def map_vects(self,input_vects):
        # maps input vects to relevant neurons in SOM grid
        # input_vects -> iterable of 1d numpy arrays w/ dimension dim
        if not self._trained:
            raise ValueError("Not trained, no vects")
        to_return = []
        for vect in input_vects:
            min_index = min(
                [i for i in range(len(self._weightages))],
                key=lambda x: np.linalg.norm(vect-self._weightages[x])
            )
            to_return.append(self._locations[min_index])
 
        return to_return


df_538 = pd.read_csv('https://raw.githubusercontent.com/fivethirtyeight/data/master/pollster-ratings/raw-polls.csv')
s_df_538 = df_538.sort_values(by=['pollster'])
s_df_538.set_index(keys=['pollster'], inplace='True')
list_pollsters = df_538.pollster.unique()
acc = df_538.groupby('pollster')['rightcall'].aggregate({'Mean Accuracy':"mean", 'Count':len}).sort_values('Count', ascending=False)
samplesize = df_538.groupby('pollster')['samplesize'].aggregate({'Mean Sample Size':"mean", 'Count':len}).sort_values('Count', ascending=False)
error = df_538.groupby('pollster')['error'].aggregate({'Mean Error':"mean", 'Count':len}).sort_values('Count', ascending=False)
bias= df_538.groupby('pollster')['bias'].aggregate({'Mean Bias':"mean", 'Count':len}).sort_values('Count', ascending=False)
# Put all this stuff in one dataframe
aggg = acc
aggg['Mean Sample Size'] = samplesize['Mean Sample Size']
aggg['Mean Error'] = error['Mean Error']
aggg['Mean Bias'] = bias['Mean Bias']
aggg.fillna(value=0, inplace=True)
aggg = aggg[aggg['Count'] > 10]
# Scale the data appropriately such that all axis are on similar ranges
aggg['Adjusted Mean Accuracy'] = acc['Mean Accuracy'] * 10
aggg['Adjusted Mean Sample Size'] = samplesize['Mean Sample Size'] / 60000
aggg['Adjusted Mean Error'] = error['Mean Error']
aggg['Adjusted Mean Bias'] = bias['Mean Bias'].map(abs)
aggg.fillna(value=0, inplace=True)
# filter out the really crappy ones
aggg = aggg[aggg['Count'] > 10]
cluster_this = aggg[['Adjusted Mean Accuracy', 'Adjusted Mean Sample Size', 'Adjusted Mean Error', 'Adjusted Mean Bias']]

som = TF_SOM(200,1,4,1000)
som.train(cluster_this.values)
mapped = som.map_vects(cluster_this.values)
cpp['som_score'] = [x[0] for x in mapped]
sort_som = cpp.sort_values(by='som_score')

m,b = np.polyfit(sort_som['som_score'], sort_som['Mean Accuracy'],1)
if m < 0:
    sort_som['som_score'] = sort_som['som_score'].apply(lambda x:xdim - x)
    sort_som = sort_som.sort_values(by='som_score')

fp = open('pollsters_df.dill')
dill.dump(sort_som, fp)
fp.close()

