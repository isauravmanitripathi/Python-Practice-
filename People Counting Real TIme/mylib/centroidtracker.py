# import the necessary packages 
from ast import Or
from cmath import rect
from scipy.spatial import distance as dist 
from collections import OrderedDict
import numpy as np 

class CentroidTracker:
    def __init__(self, maxDisappeared = 50, maxDistance = 50):
        #initialize the next unique object ID along with two ordered dictionaires 
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consectuive frames a given object is allowed to be marked
        self.maxDisappeared = maxDisappeared

        #store the maximum distane between centroids to associated 
        self.maxDistance = maxDistance

    def register(self, centroid):
        # when registering an object we use the next available object 
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, ObjectID):
        #to deregister an object ID we delete the object ID from both of our repsective dictionaries 
        del self.objects[ObjectID]
        del self.disappeared[ObjectID]


    def update(self, rects):
        # check to see if the list of input bounding box rectangles is empty 

        if len(rects) == 0:
            #loop over any existing tracked objects and mark them as disappeared 
            for ObjectID in list(self.disappeared.keys()):
                self.disappeared[ObjectID] += 1

                # if we have reached a maximum number of consectuive frames where a given object has been markd as missing 
                if self.disappeared[ObjectID] > self.maxDisappeared:
                    self.deregister(ObjectID)

            return self.objects

        inputCentroids = np.zeros((len(rect), 2), dtype = "init")

        #loop over the bounding box rectangles 
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endY) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        

        else:
            ObjectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())


            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis = 1).argsort()

            cols = D.argmin(axis = 1)[rows]

            usedRows = set()
            usedCols = set()

            for(row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.maxDistance:
                    continue

                objectID = ObjectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[0])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = ObjectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else: 
                for col in unusedCols:
                    self.register(inputCentroids[col])
        
        return self.objects