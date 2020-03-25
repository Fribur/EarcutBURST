using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Burst;
using Unity.Mathematics;
using UnityEngine;

public class NativeTesselateJob
{
    [BurstCompile(CompileSynchronously = true)]
    public struct EarcutJob : IJob
    {
        [ReadOnly] public NativeArray<int> FeatureIndex; //need to identify upon job complettion which polygon was tesselated
        [ReadOnly] public NativeArray<float2> InputContours; //linear Polygon including all holes
        [ReadOnly] public NativeArray<int> holeIndices; //indices of holes
        public NativeList<int> IndexOutput; //Triangle indexes of tesselated mesh
        public NativeArray<float3> normalsOutput; //Surface normals of tesselated mesh

        public void Execute()
        {
            Tessellate(InputContours, holeIndices, IndexOutput, normalsOutput);
        }
    }
    [BurstCompile(CompileSynchronously = true)]
    public struct nativeLL
    {
        public NativeList<Node> NodeLists;
        public NativeList<int> PrevLists;
        public NativeList<int> NextLists;
        public NativeList<int> PrevZLists;
        public NativeList<int> NextZLists;
    }

    public static void Tessellate(NativeArray<float2> data, NativeArray<int> holeIndices, NativeList<int> triangles, NativeArray<float3> normals)
    {
        bool hasHoles = holeIndices.Length > 0;
        int outerLen = hasHoles ? holeIndices[0] : data.Length;

        nativeLL oLL = new nativeLL();
        oLL.NodeLists = new NativeList<Node>(Allocator.Temp);
        oLL.PrevLists = new NativeList<int>(Allocator.Temp);
        oLL.NextLists = new NativeList<int>(Allocator.Temp);
        oLL.PrevZLists = new NativeList<int>(Allocator.Temp);
        oLL.NextZLists = new NativeList<int>(Allocator.Temp);

        var outerNode = LinkedList(oLL, data, 0, outerLen, true);

        int prev = oLL.PrevLists[outerNode];
        int next = oLL.NextLists[outerNode];
        if (Equals(oLL.NodeLists[next], oLL.NodeLists[prev]))
        {
            return;
        }

        var minX = double.PositiveInfinity;
        var minY = double.PositiveInfinity;
        var maxX = double.NegativeInfinity;
        var maxY = double.NegativeInfinity;
        var invSize = default(double);

        if (hasHoles)
        {
            outerNode = EliminateHoles(oLL, data, holeIndices);
        }
        // if the shape is not too simple, we'll use z-order curve hash later; calculate polygon bbox
        if (data.Length > 80)
        {
            double x, y;
            for (int i = 0; i < outerLen; i++)
            {
                x = data[i].x;
                y = data[i].y;
                if (x < minX)
                    minX = x;
                if (y < minY)
                    minY = y;
                if (x > maxX)
                    maxX = x;
                if (y > maxY)
                    maxY = y;
            }

            // minX, minY and invSize are later used to transform coords into integers for z-order calculation
            invSize = math.max(maxX - minX, maxY - minY);
            invSize = invSize != 0 ? 1 / invSize : 0;
        }        

        EarcutLinked(oLL, outerNode, triangles, minX, minY, invSize, 0);
        CalculateNormals(data, triangles, normals);
        return;
    }
       
    // Creates a circular doubly linked list from polygon points in the specified winding order.
    static int LinkedList(nativeLL aLL, NativeArray<float2> data, int start, int end, bool clockwise)
    {

        //var last = default(int);

        if (clockwise == SignedArea(data, start, end))
        {
            for (int i = start; i < end; i++)
                AddNode(aLL, i, data[i].x, data[i].y);
        }
        else
        {
            for (int i = end - 1; i >= start; i--)
                AddNode(aLL, i, data[i].x, data[i].y);

        }
        int last = aLL.NodeLists.Length - 1;
        if (Equals(aLL.NodeLists[0], aLL.NodeLists[last]))
        {
            RemoveNode(aLL, last);
        }
        return 0;
    }

    // eliminate colinear or duplicate points
    static int FilterPoints(nativeLL aLL, int startID, int endID = -1)
    {
        if (startID == -1)
        {
            return startID;
        }

        if (endID == -1)
        {
            endID = startID;
        }

        int pID = startID;
        int pNextID = aLL.NextLists[pID];
        int pPrevID = aLL.PrevLists[pID];
        bool again;


        Node p, pNext, pPrev;

        do
        {
            again = false;
            p = aLL.NodeLists[pID];
            pNextID = aLL.NextLists[pID];
            pPrevID = aLL.PrevLists[pID];
            pNext = aLL.NodeLists[pNextID];
            pPrev = aLL.NodeLists[pPrevID];

            if (!p.steiner && (Equals(p, pNext) || Area(pPrev, p, pNext) == 0))
            {
                RemoveNode(aLL, pID);
                pID = endID = pPrevID;
                if (pID == pNextID)
                {
                    break;
                }
                again = true;

            }
            else
            {
                pID = aLL.NextLists[pID];
            }
        } while (again || pID != endID);

        return endID;
    }
    
    // main ear slicing loop which triangulates a polygon (given as a linked list)
    static void EarcutLinked(nativeLL aLL, int earID, NativeList<int> triangles, double minX, double minY, double invSize, int pass = 0)
    {  
        if (earID == -1)
        {
            return;
        }

        // interlink polygon nodes in z-order
        if (pass == 0 && invSize != 0)
        {
            IndexCurve(aLL, earID, minX, minY, invSize);
        }
                
        var stopID = earID;
        Node ear, prev, next;
        int earNextID = aLL.NextLists[earID];
        int earPrevID = aLL.PrevLists[earID];

        // iterate through ears, slicing them one by one
        while (earPrevID != earNextID)
        {
            ear = aLL.NodeLists[earID];
            earNextID = aLL.NextLists[earID];
            earPrevID = aLL.PrevLists[earID];
            prev = aLL.NodeLists[earPrevID];
            next = aLL.NodeLists[earNextID];

            if (invSize != 0 ? IsEarHashed(aLL, earID, minX, minY, invSize) : IsEar(aLL, earID))
            {
                // cut off the triangle
                triangles.Add(prev.i);
                triangles.Add(ear.i);
                triangles.Add(next.i);

                RemoveNode(aLL, earID);
   
                // skipping the next vertex leads to less sliver triangles
                earID = aLL.NextLists[earNextID];
                stopID = aLL.NextLists[earNextID];
                earNextID = aLL.NextLists[earID];
                earPrevID = aLL.PrevLists[earID];
                
                continue;
            }
            earID = earNextID;
            earNextID = aLL.NextLists[earID];
            earPrevID = aLL.PrevLists[earID];

            // if we looped through the whole remaining polygon and can't find any more ears
            if (earID == stopID)
            {
                // try filtering points and slicing again
                if (pass == 0)
                {
                    EarcutLinked(aLL, FilterPoints(aLL, earID), triangles, minX, minY, invSize, 1);

                    // if this didn't work, try curing all small self-intersections locally
                }
                else if (pass == 1)
                {                    
                    EarcutLinked(aLL, earID, triangles, minX, minY, invSize, 2);

                    // as a last resort, try splitting the remaining polygon into two
                }
                else if (pass == 2)
                {
                    SplitEarcut(aLL, earID, triangles, minX, minY, invSize);
                }

                break;
            }
        }
    }

    // check whether a polygon node forms a valid ear with adjacent nodes
    static bool IsEar(nativeLL aLL, int earID)
    {
        int prevID = aLL.PrevLists[earID];
        int nextID = aLL.NextLists[earID];


        Node a = aLL.NodeLists[prevID];
        Node b = aLL.NodeLists[earID];
        Node c = aLL.NodeLists[nextID];

        if (Area(a, b, c) >= 0)
        {
            return false; // reflex, can't be an ear
        }
        // now make sure we don't have other points inside the potential ear
        int pID = aLL.NextLists[nextID];
        int pPrevID = aLL.PrevLists[pID];
        int pNextID;

        Node p, pPrev, pNext;

        while (pID != prevID)
        {
            p = aLL.NodeLists[pID];
            pNextID = aLL.NextLists[pID];
            pPrevID = aLL.PrevLists[pID];
            pPrev = aLL.NodeLists[pPrevID];
            pNext = aLL.NodeLists[pNextID];
            if (PointInTriangle(a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y) &&
                Area(pPrev, p, pNext) >= 0)
            {
                return false;
            }
            pID = aLL.NextLists[pID];
        }
        return true;
    }

    static bool IsEarHashed(nativeLL aLL, int earID, double minX, double minY, double invSize)
    {
        int prevID = aLL.PrevLists[earID];
        int nextID = aLL.NextLists[earID];

        Node a = aLL.NodeLists[prevID];
        Node b = aLL.NodeLists[earID];
        Node c = aLL.NodeLists[nextID];

        if (Area(a, b, c) >= 0)
        {
            return false; // reflex, can't be an ear
        }

        // triangle bbox; min & max are calculated like this for speed
        var minTX = a.x < b.x ? (a.x < c.x ? a.x : c.x) : (b.x < c.x ? b.x : c.x);
        var minTY = a.y < b.y ? (a.y < c.y ? a.y : c.y) : (b.y < c.y ? b.y : c.y);
        var maxTX = a.x > b.x ? (a.x > c.x ? a.x : c.x) : (b.x > c.x ? b.x : c.x);
        var maxTY = a.y > b.y ? (a.y > c.y ? a.y : c.y) : (b.y > c.y ? b.y : c.y);

        // z-order range for the current triangle bbox;
        var minZ = ZOrder(minTX, minTY, minX, minY, invSize);
        var maxZ = ZOrder(maxTX, maxTY, minX, minY, invSize);

        // look for points inside the triangle in both directions
        int pPrevID, pNextID, nPrevID, nNextID;
        Node p, pPrev, pNext, n, nPrev, nNext;

        int pID = aLL.PrevZLists[earID];
        int nID = aLL.NextZLists[earID];
        if (nID != -1)
            n = aLL.NodeLists[nID];
        else
            n = default;
        if (pID != -1)
            p = aLL.NodeLists[pID];
        else
            p = default;

        while (pID != -1 && p.z >= minZ && nID != -1 && n.z <= maxZ)
        {
            pPrevID = aLL.PrevLists[pID];
            pNextID = aLL.NextLists[pID];
            pPrev = aLL.NodeLists[pPrevID];
            pNext = aLL.NodeLists[pNextID];

            if (!Equals(p, a) && !Equals(p, c) &&
                PointInTriangle(a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y) &&
                Area(pPrev, p, pNext) >= 0)
            {
                return false;
            }
            pID = aLL.PrevZLists[pID];
            if (pID != -1)
                p = aLL.NodeLists[pID];

            nPrevID = aLL.PrevLists[nID];
            nNextID = aLL.NextLists[nID];
            nPrev = aLL.NodeLists[nPrevID];
            nNext = aLL.NodeLists[nNextID];
            if (!Equals(n, a) && !Equals(n, c) &&
                PointInTriangle(a.x, a.y, b.x, b.y, c.x, c.y, n.x, n.y) &&
                Area(nPrev, n, nNext) >= 0)
            {
                return false;
            }
            nID = aLL.NextZLists[nID];
            if (nID != -1)
                n = aLL.NodeLists[nID];
        }


        // look for remaining points in decreasing z-order
        while (pID != -1 && p.z >= minZ)
        {
            pPrevID = aLL.PrevLists[pID];
            pNextID = aLL.NextLists[pID];
            pPrev = aLL.NodeLists[pPrevID];
            pNext = aLL.NodeLists[pNextID];
            if (!Equals(p, a) && !Equals(p, c) &&
                PointInTriangle(a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y) &&
                Area(pPrev, p, pNext) >= 0)
            {
                return false;
            }
            pID = aLL.PrevZLists[pID];
            if (pID != -1)
                p = aLL.NodeLists[pID];
        }

        // look for remaining points in increasing z-order
        while (nID != -1 && n.z <= maxZ)
        {
            nPrevID = aLL.PrevLists[nID];
            nNextID = aLL.NextLists[nID];
            nPrev = aLL.NodeLists[nPrevID];
            nNext = aLL.NodeLists[nNextID];

            if (!Equals(n, a) && !Equals(n, c) &&
                PointInTriangle(a.x, a.y, b.x, b.y, c.x, c.y, n.x, n.y) &&
                Area(nPrev, n, nNext) >= 0)
            {
                return false;
            }
            nID = aLL.NextZLists[nID];
            if (nID != -1)
                n = aLL.NodeLists[nID];
        }


        return true;
    }

    // go through all polygon nodes and cure small local self-intersections
    static int CureLocalIntersections(nativeLL aLL, int startID, NativeList<int> triangles)
    {
        int pID = startID;
        int aID, bID, pNextID;
        Node a, b, p, pNext;

        do
        {
            aID = aLL.PrevLists[pID];
            pNextID = aLL.NextLists[pID];
            bID = aLL.NextLists[pNextID];
            a = aLL.NodeLists[aID];
            b = aLL.NodeLists[bID];
            p = aLL.NodeLists[pID];
            pNext = aLL.NodeLists[pNextID];

            if (!Equals(a, b) && Intersects(a, p, pNext, b) && LocallyInside(aLL, aLL, aID, bID) && LocallyInside(aLL, aLL, bID, aID))
            {

                triangles.Add(a.i);
                triangles.Add(p.i);
                triangles.Add(b.i);

                // remove two nodes involved
                RemoveNode(aLL, pID);
                RemoveNode(aLL, pNextID);

                pID = startID = bID;
            }
            pID = aLL.NextLists[pID];
        } while (pID != startID);

        return FilterPoints(aLL, pID);
    }

    // try splitting polygon into two and triangulate them independently
    static void SplitEarcut(nativeLL aLL, int startID, NativeList<int> triangles, double minX, double minY, double invSize)
    {
        // look for a valid diagonal that divides the polygon into two
        int aID = startID;
        int aNextID, bID, aPrevID, cID, cNextID;
        Node a, b;
        do
        {
            aPrevID = aLL.PrevLists[aID];
            aNextID = aLL.NextLists[aID];
            bID = aLL.NextLists[aNextID];

            while (bID != aPrevID)
            {
                a = aLL.NodeLists[aID];
                b = aLL.NodeLists[bID];
                if (a.i != b.i && IsValidDiagonal(aLL, aLL, aID, bID))
                {
                    // split the polygon in two by the diagonal
                    cID = SplitPolygon(aLL, aLL, aID, bID);

                    // filter colinear points around the cuts
                    aNextID = aLL.NextLists[aID];
                    aID = FilterPoints(aLL, aID, aNextID);

                    cNextID = aLL.NextLists[cID];
                    cID = FilterPoints(aLL, cID, cNextID);

                    // run earcut on each half
                    EarcutLinked(aLL, aID, triangles, minX, minY, invSize);
                    EarcutLinked(aLL, cID, triangles, minX, minY, invSize);
                    return;
                }
                bID = aLL.NextLists[bID];
            }
            aID = aLL.NextLists[aID];
        } while (aID != startID);
    }
    [BurstCompile]
    struct PolygonPOI
    {
        public Node Point;
        public int poiID;
        public int PolygonID;
        public PolygonPOI(Node _point, int _poiID, int _polygonID)
        {
            this.Point = _point;
            this.poiID = _poiID;
            this.PolygonID = _polygonID;
        }
    }

    // link every hole into the outer loop, producing a single-ring polygon without holes
    static int EliminateHoles(nativeLL oLL, NativeArray<float2> data, NativeArray<int> holeIndices)
    {
        NativeList<PolygonPOI> queue = new NativeList<PolygonPOI>(Allocator.Temp);
        int outerNode = 0;
        float2 leftmost = new float2();

        var len = holeIndices.Length;

        for (var i = 0; i < len; i++)
        {
            var start = holeIndices[i];
            var end = i < len - 1 ? holeIndices[i + 1] : data.Length;
            int leftMostID = 0;
            leftmost = data[start];
            for (int k = start; k < end; k++)
            {
                if (data[k].x < leftmost.x || (data[k].x == leftmost.x && data[k].y < leftmost.y))
                {
                    leftMostID = k - start;
                    leftmost = data[k];
                }
            }
            queue.Add(new PolygonPOI(new Node(leftMostID + start, leftmost.x, leftmost.y), leftMostID, i));
        }
        CompareX compareX = new CompareX();
        queue.Sort(compareX);

        nativeLL hLL = new nativeLL();
        hLL.NodeLists = new NativeList<Node>(Allocator.Temp);
        hLL.PrevLists = new NativeList<int>(Allocator.Temp);
        hLL.NextLists = new NativeList<int>(Allocator.Temp);
        hLL.PrevZLists = new NativeList<int>(Allocator.Temp);
        hLL.NextZLists = new NativeList<int>(Allocator.Temp);

        // process holes from left to right
        for (var i = 0; i < queue.Length; i++)
        {
            int start = holeIndices[queue[i].PolygonID];
            var end = queue[i].PolygonID < holeIndices.Length - 1 ? holeIndices[queue[i].PolygonID + 1] : data.Length;
            if (queue[i].PolygonID >= holeIndices.Length - 1)
            { }
            var firstNode = LinkedList(hLL, data, start, end, false);
            int nextID = hLL.NextLists[firstNode];
            if (firstNode == nextID)
            {
                var temp = hLL.NodeLists[firstNode];
                temp.steiner = true;
                hLL.NodeLists[firstNode] = temp;
            }
            var test1 = queue[i].poiID;
            EliminateHole(oLL, hLL, test1, outerNode);

            int outerNodeNextID = oLL.NextLists[outerNode];
            outerNode = FilterPoints(oLL, outerNode, outerNodeNextID);
            hLL.NodeLists.Clear();
            hLL.PrevLists.Clear();
            hLL.NextLists.Clear();
        }
        return outerNode;
    }

    struct CompareX : IComparer<PolygonPOI>
    {
        public int Compare(PolygonPOI a, PolygonPOI b)
        {
            if (a.Point.x > b.Point.x) //sort list accending
                return 1;
            else if (a.Point.x < b.Point.x)
                return -1;
            else
                return 0;
        }
    }
    
    // find a bridge between vertices that connects hole with an outer ring and and link it
    static void EliminateHole(nativeLL oLL, nativeLL hLL, int holeLeftmostID, int outerNodeID)
    {

        outerNodeID = FindHoleBridge(oLL, hLL, holeLeftmostID, outerNodeID);
        if (outerNodeID != -1)
        {
            var bID = SplitPolygon(oLL, hLL, outerNodeID, holeLeftmostID);
            //// filter collinear points around the cuts
            int OuterPolgonBridgeNodeNextID = oLL.NextLists[outerNodeID];
            FilterPoints(oLL, outerNodeID, OuterPolgonBridgeNodeNextID);
            var bNextID = oLL.NextLists[bID];
            int te = FilterPoints(oLL, bID, bNextID);     
        }
    }

    // David Eberly's algorithm for finding a bridge between hole and outer polygon
    static int FindHoleBridge(nativeLL oLL, nativeLL hLL, int holeLeftmostID, int outerNodeID)
    {        
        int nextID;
        int pID = outerNodeID;
        Node p, pNext, m;
        var hx = hLL.NodeLists[holeLeftmostID].x;
        var hy = hLL.NodeLists[holeLeftmostID].y;
        var qx = double.NegativeInfinity;
        int mID = -1;

        // find a segment intersected by a ray from the hole's leftmost point to the left;
        // segment's endpoint with lesser x will be potential connection point
        do
        {
            p = oLL.NodeLists[pID];
            nextID = oLL.NextLists[pID];
            pNext = oLL.NodeLists[nextID];
            if (p.i == 8511)
            { }

            //When testing outer polygone p clockwise, then first bridge candiate to hole h has py < hy and next py > hy
            //otherwise py > hy and next py < hy. Will not happen, as LinkedList is constructed such that outer polygone = clockwise
            if (hy >= p.y && hy <= pNext.y && pNext.y != p.y)
            {
                var x = p.x + (hy - p.y) * (pNext.x - p.x) / (pNext.y - p.y);
                if (x <= hx && x > qx)
                {
                    qx = x;
                    if (x == hx)
                    {
                        if (hy == p.y)
                        {
                            return pID;
                        }

                        if (hy == pNext.y)
                        {
                            return nextID;
                        }
                    }
                    mID = p.x < pNext.x ? pID : nextID;
                }
            }
            pID = oLL.NextLists[pID];
        } while (pID != outerNodeID);


        if (mID == -1)
        {
            return -1;
        }

        if (hx == qx)
        {
            return mID; // hole touches outer segment; pick leftmost endpoint
        }

        // look for points inside the triangle of hole point, segment intersection and endpoint;
        // if there are no points found, we have a valid connection;
        // otherwise choose the point of the minimum angle with the ray as connection point

        var stop = mID;
        m = oLL.NodeLists[mID];
        var mx = m.x;
        var my = m.y;
        var tanMin = double.PositiveInfinity;
        double tan;

        pID = mID;

        do
        {
            p = oLL.NodeLists[pID];
            m = oLL.NodeLists[mID];
            if (hx >= p.x && p.x >= mx && hx != p.x && PointInTriangle(hy < my ? hx : qx, hy, mx, my, hy < my ? qx : hx, hy, p.x, p.y))
            {

                tan = math.abs(hy - p.y) / (hx - p.x); // tangential

                if (LocallyInside(oLL, hLL, pID, holeLeftmostID) && (tan < tanMin || (tan == tanMin && p.x > m.x || (p.x == m.x && sectorContainsSector(oLL, oLL, mID, pID)))))
                {
                    mID = pID;
                    tanMin = tan;
                }
            }

            pID = oLL.NextLists[pID];
        } while (pID != stop);

        return mID;
    }

    // interlink polygon nodes in z-order
    static void IndexCurve(nativeLL aLL, int startID, double minX, double minY, double invSize)
    {
        int pID = startID;
        Node p;
        do
        {
            p = aLL.NodeLists[pID];
            if (p.z == -1)
            {
                p.z = ZOrder(p.x, p.y, minX, minY, invSize);
                aLL.NodeLists[pID] = p;
            }
            aLL.PrevZLists[pID] = aLL.PrevLists[pID];
            aLL.NextZLists[pID] = aLL.NextLists[pID];
            pID = aLL.NextLists[pID];
        } while (pID != startID);
        int pPrevZID = aLL.PrevZLists[pID];
        aLL.NextZLists[pPrevZID] = -1;
        aLL.PrevZLists[pID] = -1;
        SortLinked(aLL, pID);
    }

    // Simon Tatham's linked list merge sort algorithm
    // http://www.chiark.greenend.org.uk/~sgtatham/algorithms/listsort.html
    static int SortLinked(nativeLL aLL, int headID)
    {
        int i;
        int pID, pNextZID, qID, qNextZID, eID, tailID;
        qNextZID = -1;
        Node p;
        Node q;
        int numMerges;
        int pSize;
        int qSize;
        int inSize = 1;

        do
        {
            pID = headID;
            headID = -1;
            tailID = -1;
            numMerges = 0;

            while (pID != -1)
            {
                numMerges++;
                qID = pID;
                pSize = 0;
                for (i = 0; i < inSize; i++)
                {
                    pSize++;
                    qID = aLL.NextZLists[qID];
                    if (qID == -1)
                    {
                        break;
                    }
                }
                qSize = inSize;

                while (pSize > 0 || (qSize > 0 && qID != -1))
                {
                    if (qID != -1)
                    {
                        q = aLL.NodeLists[qID];
                        qNextZID = aLL.NextZLists[qID];
                    }
                    else
                        q = default;
                    p = aLL.NodeLists[pID];
                    pNextZID = aLL.NextZLists[pID];
                    if (pSize != 0 && (qSize == 0 || qID == -1 || p.z <= q.z))
                    {
                        eID = pID;
                        pID = pNextZID;
                        pSize--;
                    }
                    else
                    {
                        eID = qID;
                        qID = qNextZID;
                        qSize--;
                    }

                    if (tailID != -1)
                    {
                        aLL.NextZLists[tailID] = eID;
                    }
                    else
                        headID = eID;
                    aLL.PrevZLists[eID] = tailID;
                    tailID = eID;
                }
                pID = qID;
            }
            if (tailID != -1)
                aLL.NextZLists[tailID] = -1;
            inSize *= 2;

        } while (numMerges > 1);        
        return headID;
    }

    // z-order of a point given coords and inverse of the longer side of data bbox
    static int ZOrder(double x, double y, double minX, double minY, double invSize)
    {
        // coords are transformed into non-negative 15-bit integer range
        int intX = (int)(32767 * (x - minX) * invSize);
        int intY = (int)(32767 * (y - minY) * invSize);

        intX = (intX | (intX << 8)) & 0x00FF00FF;
        intX = (intX | (intX << 4)) & 0x0F0F0F0F;
        intX = (intX | (intX << 2)) & 0x33333333;
        intX = (intX | (intX << 1)) & 0x55555555;

        intY = (intY | (intY << 8)) & 0x00FF00FF;
        intY = (intY | (intY << 4)) & 0x0F0F0F0F;
        intY = (intY | (intY << 2)) & 0x33333333;
        intY = (intY | (intY << 1)) & 0x55555555;

        return intX | (intY << 1);
    }

    // find the leftmost node of a polygon ring
    static int GetLeftmost(nativeLL aLL)
    {
        int pID = 0;
        int leftmostID = 0;
        Node p;
        Node leftmost;
        do
        {
            p = aLL.NodeLists[pID];
            leftmost = aLL.NodeLists[leftmostID];
            if (p.x < leftmost.x || (p.x == leftmost.x && p.y < leftmost.y))
            {
                leftmostID = pID;
            }
            pID = aLL.NextLists[pID];
        } while (pID != 0);

        return leftmostID;
    }

    // DOT product test (method 3 here: http://totologic.blogspot.com/2014/01/accurate-point-in-triangle-test.html
    //needs to work regardless of tringle winding, see here http://www.sunshine2k.de/coding/java/PointInTriangle/PointInTriangle.html
    static bool PointInTriangle(double ax, double ay, double bx, double by, double cx, double cy, double px, double py)
    {
        double side1 = (cx - px) * (ay - py) - (ax - px) * (cy - py);
        double side2 = (ax - px) * (by - py) - (bx - px) * (ay - py);
        double side3 = (bx - px) * (cy - py) - (cx - px) * (by - py);
        return (side1 > 0 && side2 > 0 && side3 > 0) || (side1 < 0 && side2 < 0 && side3 < 0); //first test works for CCW-triangles, second for CW-triangles;
    }

    // check if a diagonal between two polygon nodes is valid (lies in polygon interior)
    static bool IsValidDiagonal(nativeLL aLL, nativeLL bLL, int aID, int bID)
    {
        int aNextID = aLL.NextLists[aID];
        int aPrevID = aLL.PrevLists[aID];
        int bNextID = bLL.NextLists[bID];
        int bPrevID = bLL.PrevLists[bID];
        Node a = aLL.NodeLists[aID];
        Node aNext = aLL.NodeLists[aNextID];
        Node aPrev = aLL.NodeLists[aPrevID];
        Node b = bLL.NodeLists[bID];
        Node bNext = bLL.NodeLists[bNextID];
        Node bPrev = bLL.NodeLists[bPrevID];

        return aNext.i != b.i && aPrev.i != b.i && !IntersectsPolygon(aLL, bLL, aID, bID) && // dones't intersect other edges
                (LocallyInside(aLL, bLL, aID, bID) && LocallyInside(aLL, bLL, bID, aID) && MiddleInside(aLL, bLL, aID, bID) && // locally visible
                (Area(aPrev, a, bPrev) != 0 || Area(a, bPrev, b) != 0) || // does not create opposite-facing sectors
                Equals(a, b) && Area(aPrev, a, aNext) > 0 && Area(bPrev, b, bNext) > 0); // special zero-length case
    }

    //signed area of a triangle
    static double Area(Node p, Node q, Node r) //CCW triangle = positive area, CC triangle = negative area
    {
        return (q.x - p.x) * (r.y - q.y) - (r.x - q.x) * (q.y - p.y); //https://algs4.cs.princeton.edu/91primitives/;
    }

    //// check if two points are equal
    static bool Equals(Node p1, Node p2)
    {
        return p1.x == p2.x && p1.y == p2.y;
    }

    // check if two segments intersect
    static bool Intersects(Node p1, Node q1, Node p2, Node q2)
    {
        var o1 = Sign(Area(p1, q1, p2));
        var o2 = Sign(Area(p1, q1, q2));
        var o3 = Sign(Area(p2, q2, p1));
        var o4 = Sign(Area(p2, q2, q1));

        if (o1 != o2 && o3 != o4) return true; // general case

        if (o1 == 0 && OnSegment(p1, p2, q1)) return true; // p1, q1 and p2 are collinear and p2 lies on p1q1
        if (o2 == 0 && OnSegment(p1, q2, q1)) return true; // p1, q1 and q2 are collinear and q2 lies on p1q1
        if (o3 == 0 && OnSegment(p2, p1, q2)) return true; // p2, q2 and p1 are collinear and p1 lies on p2q2
        if (o4 == 0 && OnSegment(p2, q1, q2)) return true; // p2, q2 and q1 are collinear and q1 lies on p2q2

        return false;
    }

    // for collinear points p, q, r, check if point q lies on segment pr
    static bool OnSegment(Node p, Node q, Node r)
    {
        return q.x <= math.max(p.x, r.x) && q.x >= math.min(p.x, r.x) && q.y <= math.max(p.y, r.y) && q.y >= math.min(p.y, r.y);
    }

    static int Sign(double num)
    {
        return num > 0 ? 1 : num < 0 ? -1 : 0;
    }

    // check if a polygon diagonal intersects any polygon segments
    static bool IntersectsPolygon(nativeLL aLL, nativeLL bLL, int aID, int bID)
    {
        int aNextID = aLL.NextLists[aID];
        int pID = aID;
        int aPrevID = aLL.PrevLists[aID];
        int pNextID;
        Node a = aLL.NodeLists[aID];
        Node aNext = aLL.NodeLists[aNextID];
        Node aPrev = aLL.NodeLists[aPrevID];
        Node b = bLL.NodeLists[bID];
        Node p = a;
        Node pNext;

        do
        {
            p = aLL.NodeLists[pID];
            pNextID = aLL.NextLists[pID];
            pNext = aLL.NodeLists[pNextID];
            if (p.i != a.i && pNext.i != a.i && p.i != b.i && pNext.i != b.i &&
                    Intersects(p, pNext, a, b))
            {
                return true;
            }
            pID = aLL.NextLists[pID];
        } while (pID != aID);
        return false;
    }

    // check if a polygon diagonal is locally inside the polygon
    static bool LocallyInside(nativeLL aLL, nativeLL bLL, int aID, int bID)
    {
        int aNextID, aPrevID;
        Node a, aNext, aPrev, b;
        aNextID = aLL.NextLists[aID];
        aPrevID = aLL.PrevLists[aID];
        a = aLL.NodeLists[aID];
        aNext = aLL.NodeLists[aNextID];
        aPrev = aLL.NodeLists[aPrevID];
        b = bLL.NodeLists[bID];
        return Area(aPrev, a, aNext) < 0 ?
            Area(a, b, aNext) >= 0 && Area(a, aPrev, b) >= 0 :
            Area(a, b, aPrev) < 0 || Area(a, aNext, b) < 0;
    }

    // whether sector in vertex m contains sector in vertex p in the same coordinates
    static bool sectorContainsSector(nativeLL aLL, nativeLL bLL, int aID, int bID)
    {
        int aNextID, aPrevID, bNextID, bPrevID;
        Node a, aNext, aPrev, bNext, bPrev;
        aNextID = aLL.NextLists[aID];
        aPrevID = aLL.PrevLists[aID];
        a = aLL.NodeLists[aID];
        aNext = aLL.NodeLists[aNextID];
        aPrev = aLL.NodeLists[aPrevID];
        bNextID = bLL.NextLists[bID];
        bPrevID = bLL.PrevLists[bID];
        bNext = bLL.NodeLists[bNextID];
        bPrev = bLL.NodeLists[bPrevID];
        return Area(aPrev, a, bPrev) < 0 && Area(bNext, a, aNext) < 0;
    }

    // check if the middle point of a polygon diagonal is inside the polygon
    static bool MiddleInside(nativeLL aLL, nativeLL bLL, int aID, int bID)
    {
        int aNextID = aLL.NextLists[aID];
        int pID = aID;
        int pNextID;
        int aPrevID = aLL.PrevLists[aID];
        Node a = aLL.NodeLists[aID];
        Node aNext = aLL.NodeLists[aNextID];
        Node aPrev = aLL.NodeLists[aPrevID];
        Node b = bLL.NodeLists[bID];
        Node p = a;
        Node pNext;

        var inside = false;
        var px = (a.x + b.x) / 2;
        var py = (a.y + b.y) / 2;
        do
        {
            p = aLL.NodeLists[pID];
            pNextID = aLL.NextLists[pID];
            pNext = aLL.NodeLists[pNextID];
            if (((p.y > py) != (pNext.y > py)) && pNext.y != p.y && (px < (pNext.x - p.x) * (py - p.y) / (pNext.y - p.y) + p.x))
            {
                inside = !inside;
            }
            pID = aLL.NextLists[pID];
        } while (pID != aID);

        return inside;
    }

    // link two polygon vertices with a bridge; if the vertices belong to the same ring, it splits polygon into two;
    // if one belongs to the outer ring and another to a hole, it merges it into a single ring
    static int SplitPolygon(nativeLL aLL, nativeLL bLL, int aID, int bID)
    {
        Node a2 = aLL.NodeLists[aID];
        Node b2 = bLL.NodeLists[bID];
        Node holeNode;
        int right = aID;
        int aNextID = aLL.NextLists[aID];
        int bPrevID = bLL.PrevLists[bID];
        int a2ID, b2ID;

        if (aLL.NodeLists.Equals(bLL.NodeLists)) ////split Polygon
        {
            //first Polygon
            aLL.NextLists[aID] = bID;
            aLL.PrevLists[bID] = aID;
            //second Polygon
            b2ID = AddNodeAfter(aLL, b2, bPrevID);
            a2ID = AddNodeBefore(aLL, a2, aNextID);
            aLL.PrevLists[a2ID] = b2ID;
            aLL.NextLists[b2ID] = a2ID;
            right = b2ID;
        }
        else //merge Polygon
        {
            int stop = bID;
            do
            {
                holeNode = bLL.NodeLists[bID];
                InsertNode(aLL, aNextID, holeNode);
                bID = bLL.NextLists[bID];
            } while (bID != stop);
            right = InsertNode(aLL, aNextID, b2);
            InsertNode(aLL, aNextID, a2);
        }

        return right;
    }

    // create a node and optionally link it with previous one (in a circular doubly linked list)
    static int AddNode(nativeLL aLL, int i, double x, double y)
    {
        aLL.NodeLists.Add(new Node(i, x, y));
        int current = aLL.NodeLists.Length - 1;
        if (current > 0)
        {
            aLL.NextLists.Add(0); //extend NextList, set Next of Tail to Head
            aLL.PrevLists.Add(current - 1); //extend PrevList, set point current index to Prev index
            aLL.NextLists[current - 1] = current;
            aLL.PrevLists[0] = current; //set Prev of Head to Tail
        }
        else
        {
            aLL.PrevLists.Add(0);
            aLL.NextLists.Add(0);
        }
        aLL.PrevZLists.Add(-1); //extend PrevZList
        aLL.NextZLists.Add(-1); //extend NextZList
        return current;
    }

    static int AddNodeBefore(nativeLL aLL, Node a, int before)
    {
        aLL.NodeLists.Add(a);
        int current = aLL.NodeLists.Length - 1;
        if (current > 0)
        {
            aLL.NextLists.Add(before); //extend NextList
            aLL.PrevLists.Add(-1); //extend PrevList, set point current index to Prev index
            aLL.PrevLists[before] = current;
            aLL.PrevZLists.Add(-1); //extend PrevZList
            aLL.NextZLists.Add(-1); //extend NextZList
        }
        return current;
    }
    static int AddNodeAfter(nativeLL aLL, Node a, int after)
    {
        aLL.NodeLists.Add(a);
        int current = aLL.NodeLists.Length - 1;
        if (current > 0)
        {
            aLL.PrevLists.Add(after); //extend PrevList
            aLL.NextLists.Add(-1); //extend NextList, set point current index to Prev index
            aLL.NextLists[after] = current;
            aLL.PrevZLists.Add(-1); //extend PrevZList
            aLL.NextZLists.Add(-1); //extend NextZList
        }
        return current;
    }


    static int InsertNode(nativeLL aLL, int aID, Node a)
    {
        aLL.NodeLists.Add(a);
        int current = aLL.NodeLists.Length - 1;
        int left = aLL.PrevLists[aID];
        aLL.NextLists[left] = current;
        aLL.PrevLists[aID] = current;
        aLL.PrevLists.Add(left);
        aLL.NextLists.Add(aID);
        aLL.PrevZLists.Add(-1); //extend PrevZList
        aLL.NextZLists.Add(-1); //extend NextZList
        return current;
    }

    static void RemoveNode(nativeLL aLL, int aID)
    {
        int prev = aLL.PrevLists[aID];
        int next = aLL.NextLists[aID];
        aLL.PrevLists[next] = prev;
        aLL.NextLists[prev] = next;
    }

    [BurstCompile(CompileSynchronously = true)]
    public struct Node
    {
        public int i;
        public double x;
        public double y;

        public int z;

        public bool steiner;

        public Node(int i, double x, double y)
        {
            // vertex index in coordinates array
            this.i = i;

            // vertex coordinates
            this.x = x;
            this.y = y;

            // z-order curve value
            this.z = -1;

            // indicates whether this is a steiner point
            this.steiner = false;
        }
    }
    static bool SignedArea(NativeArray<float2> data, int start, int end)
    {
        var sum = default(double);
        var isClockwise = false;

        for (int i = start; i < end - 1; i++)
        {
            sum += (data[i + 1].x - data[i].x) * (data[i + 1].y + data[i].y) / 2;
        }
        isClockwise = (sum > 0) ? true : false;
        return isClockwise;
    }

    static void CalculateNormals(NativeArray<float2> data, NativeList<int> triangles, NativeArray<float3> normals)
    { //shortcut: 0,1,0 for CCW polygons, 0,-1,0 for CW 
        for (int i = 0; i < triangles.Length / 3; i++)
        {
            int normalTriangleIndex = i * 3;
            int vertexIndexA = triangles[normalTriangleIndex];
            int vertexIndexB = triangles[normalTriangleIndex + 1];
            int vertexIndexC = triangles[normalTriangleIndex + 2];
            float3 AB = (new float3(data[vertexIndexB].x,0f, data[vertexIndexB].y) - new float3(data[vertexIndexA].x,0f, data[vertexIndexA].y));
            float3 AC = (new float3(data[vertexIndexC].x, 0f, data[vertexIndexC].y) - new float3(data[vertexIndexA].x, 0f, data[vertexIndexA].y));
            float3 triangleNormal = math.normalize(math.cross(AB,AC));
            normals[vertexIndexA] += triangleNormal;
            normals[vertexIndexB] += triangleNormal;
            normals[vertexIndexC] += triangleNormal;
        }
        for (int i = 0; i < normals.Length; i++)
            normals[i]=math.normalize(normals[i]);
    }
}
