import numpy

class RadialBee(object):
    """Class defines circle of a specific diameter around
    a bees central points and returns an  in a given array"""

    #---------------------------------------------------------------
    # ctor:
    # min_prob - minimum bee probability
    # max_prob - maximum bee probability
    # radius - radius of the bee circle
    #---------------------------------------------------------------
    def __init__(self, min_prob = 0.6, max_prob = 1.0, radius = 40):

        self.min_prob = min_prob
        self.max_prob = max_prob
        self.radius = radius
        self.radius_dict = dict()

        # create dictionary with x,y -> radius calculations to speed up
        # data generation
        radiusf = float(self.radius)
        for x in range(0, self.radius + 1):          
          xf = float(x)
          xf *= xf          
          for y in range(0, self.radius + 1):            
            yf = float(y)
            yf *= yf            
            bee_probability = 0.0
            current_radius = numpy.sqrt(xf + yf)
            if current_radius <= radiusf:
                # check if lies in circle
                bee_probability = self.max_prob - (self.max_prob - self.min_prob) * (current_radius / radiusf)

            self.radius_dict[(x, y)] = bee_probability

class BeesHeatMap(object):
    """Class which draws circles of a specific diameter around
    the bees central points and returns an  in a given array"""

    #---------------------------------------------------------------
    # ctor:
    # radial_bee - model of a bee
    # bees_positions_list - list with positions of bees
    # height - height of the picture
    # width - width of the picture
    #---------------------------------------------------------------
    def __init__(self, radial_bee, bees_positions_list, height, width):

        self.radial_bee = radial_bee
        self.bees_positions_list = bees_positions_list
        self.width = width
        self.height = height
            
    #---------------------------------------------------------------
    # get the heatmap for the bees as a 2 dimensional array,
    # with values from 0.0 to 1.0
    #---------------------------------------------------------------
    def get_heatmap(self):
        
        # declare result heatmap
        self.heatmap_array = numpy.zeros((self.height, self.width))
        
        # iterate over the bees
        for bee_position in self.bees_positions_list:

            # analyse each bee separately
            self.get_heatmap_per_bee(bee_position)

        return self.heatmap_array
               
    #---------------------------------------------------------------
    # get the heatmap for a bee, by filling existing 
    # 2 dimensional array, with values from 0.0 to 1.0
    # if bees overlap, then max cannot be greater than 1.0
    # heatmap_array - existing bees
    # bee_position - specific bee position
    #---------------------------------------------------------------
    def get_heatmap_per_bee(self, bee_position):
      
        x_start, y_start = bee_position

        # draw circle 
        xmin = x_start - self.radial_bee.radius
        xmin = numpy.maximum(0, xmin)
        xmax = x_start + self.radial_bee.radius
        xmax = numpy.minimum(self.width - 1, xmax)

        ymin = y_start - self.radial_bee.radius
        ymin = numpy.maximum(0, ymin)
        ymax = y_start + self.radial_bee.radius
        ymax = numpy.minimum(self.height - 1, ymax)

        for x in range(xmin, xmax):
            for y in range(ymin, ymax):

                xo = numpy.abs(x_start - x)
                yo = numpy.abs(y_start - y)
                bee_probability = self.radial_bee.radius_dict[(xo, yo)]

                self.heatmap_array[y][x] += bee_probability
                if self.heatmap_array[y][x] > 1.0:
                    self.heatmap_array[y][x] = 1.0
