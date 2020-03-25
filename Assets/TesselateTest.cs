using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;

public class TesselateTest : MonoBehaviour
{
    [SerializeField]
    Material material;
    // Start is called before the first frame update
    void Start()
    {
        string json  = File.ReadAllText("./Assets/1731.dat");
        var polygon = JsonUtility.FromJson<PolygonSerializer.FlatPolygon>(json);

        //allocate NativeArrays
        NativeArray<float2> InputContourNative = new NativeArray<float2>(polygon.PolygonData.Length, Allocator.Persistent);
        NativeArray<int> holeIndices = new NativeArray<int>(polygon.holeIndices.Length, Allocator.Persistent);
        NativeArray<int> featNative = new NativeArray<int>(1, Allocator.Persistent);
        NativeList<int> indexNative = new NativeList<int>(polygon.PolygonData.Length * 4, Allocator.Persistent);
        NativeArray<float3> normalsNative = new NativeArray<float3>(polygon.PolygonData.Length, Allocator.Persistent);

        Vector3[] vert = new Vector3[polygon.PolygonData.Length];
        float2 temp;

        //fill NativeArrays with Data
        for (int i = 0; i < polygon.PolygonData.Length; i++)
        {
            float x = polygon.PolygonData[i].x / 10000;
            float y = polygon.PolygonData[i].y / 10000;
            temp = InputContourNative[i];
            temp.x= x;
            temp.y= y;
            InputContourNative[i] = temp;
            vert[i] = new float3(x, 0, y);
        };
        holeIndices.CopyFrom(polygon.holeIndices);
        featNative[0] = 1;

        var job = new NativeTesselateJob.EarcutJob()
        {
            FeatureIndex = featNative,
            InputContours = InputContourNative,
            holeIndices = holeIndices,
            IndexOutput = indexNative,
            normalsOutput = normalsNative,
        }.Schedule();

        //RenderTesselated Polygon
        var go = new GameObject();
        go.AddComponent<MeshRenderer>().sharedMaterial = material;
        var meshFilter = go.AddComponent<MeshFilter>();
        var mesh = meshFilter.mesh;
        mesh.vertices = vert;

        job.Complete();
        mesh.SetIndices(indexNative.ToArray(), MeshTopology.Triangles,0);
        mesh.SetNormals(normalsNative);

        InputContourNative.Dispose();
        holeIndices.Dispose();
        featNative.Dispose();
        indexNative.Dispose();
        normalsNative.Dispose();
    }
}
