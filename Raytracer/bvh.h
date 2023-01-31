#include "primitive.h"
#pragma once

__declspec(align(32)) struct BVHNode
{
	float3 aabbMin, aabbMax;
	uint leftFirst, primitiveCount;
	bool isLeaf() { return primitiveCount > 0; }
};

struct GPUBVHNode
{
	float aabbMinx, aabbMiny, aabbMinz;
	float aabbMaxx, aabbMaxy, aabbMaxz;
	int leftFirst, primitiveCount;
};

__declspec(align(64)) struct SIMD_BVH_Node
{
	union { __m128 bminx4; float bminx[4]; };
	union { __m128 bmaxx4; float bmaxx[4]; };
	union { __m128 bminy4; float bminy[4]; };
	union { __m128 bmaxy4; float bmaxy[4]; };
	union { __m128 bminz4; float bminz[4]; };
	union { __m128 bmaxz4; float bmaxz[4]; };
	int child[4], count[4];
};

struct aabbSAH
{
	float3 bmin = 1e30f, bmax = -1e30f;
	void grow(float3 p) { bmin = fminf(bmin, p), bmax = fmaxf(bmax, p); }
	float area()
	{
		float3 e = bmax - bmin; // box extent
		return e.x * e.y + e.y * e.z + e.z * e.x;
	}
};

class BVH
{
public:
	int spheres_size = 0;
	Sphere* spheres = 0;

	int planes_size = 0;
	Plane* planes = 0;

    int triangles_size = 0;
    Triangle* triangles = 0;

    int* primitiveIdx = 0;
	
    BVHNode* bvhNode = 0;
	GPUBVHNode* gpuBvhNode = 0;
    uint rootNodeIdx = 0, planeRootNodeIdx = 1, primitveRootNodeIdx = 2, nodesUsed = 3;

	// QBVH
	SIMD_BVH_Node* qbvhNode = 0;
	uint nodesUsedQBVH = 0;

	// SIMD
	__m128 zero = _mm_setzero_ps();

    BVH() = default;
	BVH(Sphere* spheres, int spheres_size, Plane* planes, int planes_size, Triangle* triangles, int triangles_size ):
		spheres(spheres), spheres_size(spheres_size),
		planes(planes), planes_size(planes_size),
		triangles(triangles), triangles_size(triangles_size)
	{
        BuildBVH();
		BuildQBVH();

	}
	void CollapseBVH(int nodeIdx)
	{

		SIMD_BVH_Node& parentNode = qbvhNode[nodeIdx];
		for (int i = 0; i < 4; i++)
		{
			if (parentNode.child[i] == -1)
			{
				int greatest_child = -1;
				float greatest_area = -1.0f;

				for (int j = 0; j < i; j++)
				{
					if (parentNode.count[j] == 0)
					{
						float areaX = fabs(parentNode.bmaxx[j] - parentNode.bminx[j]);
						float areaY = fabs(parentNode.bmaxy[j] - parentNode.bminy[j]);
						float areaZ = fabs(parentNode.bmaxz[j] - parentNode.bminz[j]);

						float area = areaX * areaY + areaY * areaZ + areaZ * areaX;

						if (area > greatest_area)
						{
							greatest_area = area;
							greatest_child = j;
						}
					}
				}

				if (greatest_child == -1)  // If there is a child that we can collapse upon
					break;

				int left_child = parentNode.child[greatest_child];
				int right_child = parentNode.child[greatest_child] + 1;
				

				// Left Child
					
				parentNode.count[greatest_child] = bvhNode[left_child].primitiveCount;
				parentNode.child[greatest_child] = bvhNode[left_child].leftFirst;

				parentNode.bminx[greatest_child] = bvhNode[left_child].aabbMin.x;
				parentNode.bminy[greatest_child] = bvhNode[left_child].aabbMin.y;
				parentNode.bminz[greatest_child] = bvhNode[left_child].aabbMin.z;
				parentNode.bmaxx[greatest_child] = bvhNode[left_child].aabbMax.x;
				parentNode.bmaxy[greatest_child] = bvhNode[left_child].aabbMax.y;
				parentNode.bmaxz[greatest_child] = bvhNode[left_child].aabbMax.z;

				// Right Child
				parentNode.count[i] = bvhNode[right_child].primitiveCount;
				parentNode.child[i] = bvhNode[right_child].leftFirst;
					
				/*
				std::cout << "greatest_child: " << greatest_child << std::endl;
				std::cout << "left_child: " << left_child << std::endl;
				std::cout << "right_child: " << right_child << std::endl;
				std::cout << "child: " << parentNode.child[i] << std::endl;
				std::cout << "count: " << parentNode.count[i] << std::endl;
				*/
				parentNode.bminx[i] = bvhNode[right_child].aabbMin.x;
				parentNode.bminy[i] = bvhNode[right_child].aabbMin.y;
				parentNode.bminz[i] = bvhNode[right_child].aabbMin.z;
				parentNode.bmaxx[i] = bvhNode[right_child].aabbMax.x;
				parentNode.bmaxy[i] = bvhNode[right_child].aabbMax.y;
				parentNode.bmaxz[i] = bvhNode[right_child].aabbMax.z;

				/*
				std::cout << parentNode.child[greatest_child] << std::endl;
				std::cout << parentNode.child[i] << std::endl;
				std::cout << "-------------" << std::endl;
				*/
				for (int k = 0; k <= i; k++)
				{
					//std::cout << parentNode.child[k] << std::endl;
				}
				

			}
		}

		for (int i = 0; i < 4; i++)
		{
			if (parentNode.child[i] != -1 && parentNode.count[i] == 0) // If child node exist and is not leaf, then recurse
			{
				// Create new node
				uint newNode = nodesUsedQBVH++;


				int childrenIdx = parentNode.child[i];
				parentNode.child[i] = newNode;


				// Assign the two children to the new node of QBVH
				AssignNodeQBVH(newNode, childrenIdx);
				/*
				for (int k = 0; k < 4; k++)
				{
					std::cout << qbvhNode[newNode].child[k] << std::endl;
				}
				std::cout << "----------" << std::endl;
				*/
				CollapseBVH(newNode);
			}
		}
	}


	void AssignNodeQBVH(uint parentIdx, int childrenIdx)
	{
		SIMD_BVH_Node& parent = qbvhNode[parentIdx];

		int left_child = childrenIdx;
		int right_child = childrenIdx + 1;

		parent.bminx[0] = bvhNode[left_child].aabbMin.x;
		parent.bminy[0] = bvhNode[left_child].aabbMin.y;
		parent.bminz[0] = bvhNode[left_child].aabbMin.z;
		parent.bmaxx[0] = bvhNode[left_child].aabbMax.x;
		parent.bmaxy[0] = bvhNode[left_child].aabbMax.y;
		parent.bmaxz[0] = bvhNode[left_child].aabbMax.z;

		parent.bminx[1] = bvhNode[right_child].aabbMin.x;
		parent.bminy[1] = bvhNode[right_child].aabbMin.y;
		parent.bminz[1] = bvhNode[right_child].aabbMin.z;
		parent.bmaxx[1] = bvhNode[right_child].aabbMax.x;
		parent.bmaxy[1] = bvhNode[right_child].aabbMax.y;
		parent.bmaxz[1] = bvhNode[right_child].aabbMax.z;

		parent.count[0] = bvhNode[left_child].primitiveCount;
		parent.count[1] = bvhNode[right_child].primitiveCount;

		parent.child[0] = bvhNode[left_child].leftFirst;
		parent.child[1] = bvhNode[right_child].leftFirst;

		parent.child[2] = -1;
		parent.child[3] = -1;
	}

	void BuildQBVH()
	{
		qbvhNode = new SIMD_BVH_Node[(spheres_size + triangles_size) * 2 + 1];
		SIMD_BVH_Node& root = qbvhNode[nodesUsedQBVH++];

		root.bminx[0] = bvhNode[planeRootNodeIdx].aabbMin.x;
		root.bminy[0] = bvhNode[planeRootNodeIdx].aabbMin.y;
		root.bminz[0] = bvhNode[planeRootNodeIdx].aabbMin.z;
		root.bmaxx[0] = bvhNode[planeRootNodeIdx].aabbMax.x;
		root.bmaxy[0] = bvhNode[planeRootNodeIdx].aabbMax.y;
		root.bmaxz[0] = bvhNode[planeRootNodeIdx].aabbMax.z;

		root.child[0] = bvhNode[planeRootNodeIdx].leftFirst;
		root.count[0] = bvhNode[planeRootNodeIdx].primitiveCount;

		root.bminx[1] = bvhNode[primitveRootNodeIdx].aabbMin.x;
		root.bminy[1] = bvhNode[primitveRootNodeIdx].aabbMin.y;
		root.bminz[1] = bvhNode[primitveRootNodeIdx].aabbMin.z;
		root.bmaxx[1] = bvhNode[primitveRootNodeIdx].aabbMax.x;
		root.bmaxy[1] = bvhNode[primitveRootNodeIdx].aabbMax.y;
		root.bmaxz[1] = bvhNode[primitveRootNodeIdx].aabbMax.z;

		root.count[1] = bvhNode[primitveRootNodeIdx].primitiveCount;

		if (bvhNode[primitveRootNodeIdx].isLeaf())
			root.child[1] = bvhNode[primitveRootNodeIdx].leftFirst;
		else
		{
			root.child[1] = nodesUsedQBVH++;
			AssignNodeQBVH(root.child[1], bvhNode[primitveRootNodeIdx].leftFirst);
		}

		root.child[2] = -1;
		root.child[3] = -1;

		CollapseBVH(nodesUsedQBVH - 1);
	}

	void BuildBVH()
	{
		int nBvhNodes = (spheres_size + triangles_size) * 2 + 1;
		bvhNode = new BVHNode[nBvhNodes]; // 1 leaf node reserved for planes
		gpuBvhNode = new GPUBVHNode[nBvhNodes];

		primitiveIdx = new int[spheres_size + triangles_size];
		// populate triangle index array
		for (int i = 0; i < spheres_size + triangles_size; i++) primitiveIdx[i] = i;

		// assign all triangles to root node
		BVHNode& root = bvhNode[rootNodeIdx];
		root.leftFirst = 0, root.primitiveCount = 0;
		root.aabbMin = float3(-1e30f), root.aabbMax = float3(1e30f);

		BVHNode& planes = bvhNode[planeRootNodeIdx];
		planes.leftFirst = 0, planes.primitiveCount = planes_size;
		planes.aabbMin = float3(-1e30f), planes.aabbMax = float3(1e30f);

		BVHNode& primitve = bvhNode[primitveRootNodeIdx];
		primitve.leftFirst = 0, primitve.primitiveCount = spheres_size + triangles_size;

		UpdateNodeBounds(primitveRootNodeIdx);
		Subdivide(primitveRootNodeIdx);

		// Prepare GPU buffer
		for (int i = 0; i < nBvhNodes; i++)
		{
			gpuBvhNode[i].aabbMinx = bvhNode[i].aabbMin.x;
			gpuBvhNode[i].aabbMiny = bvhNode[i].aabbMin.y;
			gpuBvhNode[i].aabbMinz = bvhNode[i].aabbMin.z;
			gpuBvhNode[i].leftFirst = bvhNode[i].leftFirst;
			gpuBvhNode[i].aabbMaxx= bvhNode[i].aabbMax.x;
			gpuBvhNode[i].aabbMaxy = bvhNode[i].aabbMax.y;
			gpuBvhNode[i].aabbMaxz = bvhNode[i].aabbMax.z;
			gpuBvhNode[i].primitiveCount = bvhNode[i].primitiveCount;
		}
	}

	void UpdateNodeBounds(uint nodeIdx)
	{
		BVHNode& node = bvhNode[nodeIdx];
		node.aabbMin = float3(1e30f);
		node.aabbMax = float3(-1e30f);
		for (uint first = node.leftFirst, i = 0; i < node.primitiveCount; i++)
		{
			uint leafIdx = primitiveIdx[first + i];
			if (leafIdx < spheres_size)
			{
				Sphere& leafSphere = spheres[leafIdx];

				node.aabbMin = fminf(node.aabbMin, leafSphere.pos - leafSphere.r);
				node.aabbMax = fmaxf(node.aabbMax, leafSphere.pos + leafSphere.r);
			}
			else
			{
				Triangle& leafTri = triangles[leafIdx - spheres_size];
				node.aabbMin = fminf(node.aabbMin, leafTri.pos1),
				node.aabbMin = fminf(node.aabbMin, leafTri.pos2),
				node.aabbMin = fminf(node.aabbMin, leafTri.pos3),
				node.aabbMax = fmaxf(node.aabbMax, leafTri.pos1),
				node.aabbMax = fmaxf(node.aabbMax, leafTri.pos2),
				node.aabbMax = fmaxf(node.aabbMax, leafTri.pos3);
			}
		}
	}

	void Subdivide(uint nodeIdx)
	{
		// terminate recursion
		BVHNode& node = bvhNode[nodeIdx];

		float3 e = node.aabbMax - node.aabbMin; // extent of parent
		float parentArea = e.x * e.y + e.y * e.z + e.z * e.x;
		float parentCost = node.primitiveCount * parentArea;

		if (node.primitiveCount <= 2) return;
		// determine split axis using SAH
		int bestAxis = -1;
		float bestPos = 0, bestCost = 1e30f;
		for (int axis = 0; axis < 3; axis++) for (uint i = 0; i < node.primitiveCount; i++)
		{
			float candidatePos;
			if (primitiveIdx[node.leftFirst + i] < spheres_size)
				candidatePos = spheres[primitiveIdx[node.leftFirst + i]].pos[axis];
			else
				candidatePos = triangles[primitiveIdx[node.leftFirst + i] - spheres_size].centroid[axis];

			float cost = EvaluateSAH(node, axis, candidatePos);
			if (cost < bestCost)
				bestPos = candidatePos, bestAxis = axis, bestCost = cost;
		}

		if (bestCost >= parentCost) return;

		int axis = bestAxis;
		float splitPos = bestPos;

		/*
		float3 extent = node.aabbMax - node.aabbMin;
		int axis = 0;
		if (extent.y > extent.x) axis = 1;
		if (extent.z > extent[axis]) axis = 2;
		float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;
		*/
		// in-place partition
		int i = node.leftFirst;
		int j = i + node.primitiveCount - 1;
		while (i <= j)
		{
			if (primitiveIdx[i] < spheres_size)
			{
				if (spheres[primitiveIdx[i]].pos[axis] < splitPos)
					i++;
				else
					swap(primitiveIdx[i], primitiveIdx[j--]);
			}
			else
			{
				if (triangles[primitiveIdx[i] - spheres_size].centroid[axis] < splitPos)
					i++;
				else
					swap(primitiveIdx[i], primitiveIdx[j--]);
			}
		}
		// abort split if one of the sides is empty
		int leftCount = i - node.leftFirst;
		if (leftCount == 0 || leftCount == node.primitiveCount) return;
		// create child nodes
		int leftChildIdx = nodesUsed++;
		int rightChildIdx = nodesUsed++;
		bvhNode[leftChildIdx].leftFirst = node.leftFirst;
		bvhNode[leftChildIdx].primitiveCount = leftCount;
		bvhNode[rightChildIdx].leftFirst = i;
		bvhNode[rightChildIdx].primitiveCount = node.primitiveCount - leftCount;
		node.leftFirst = leftChildIdx;
		node.primitiveCount = 0;
		UpdateNodeBounds(leftChildIdx);
		UpdateNodeBounds(rightChildIdx);
		// recurse
		Subdivide(leftChildIdx);
		Subdivide(rightChildIdx);
	}

	float EvaluateSAH(BVHNode& node, int axis, float pos)
	{
		// determine triangle counts and bounds for this split candidate
		aabbSAH leftBox, rightBox;
		int leftCount = 0, rightCount = 0;
		for (uint i = 0; i < node.primitiveCount; i++)
		{
			if (primitiveIdx[node.leftFirst + i] < spheres_size)
			{
				Sphere& sphere = spheres[primitiveIdx[node.leftFirst + i]];
				if (sphere.pos[axis] < pos)
				{
					leftCount++;
					leftBox.grow(sphere.pos - sphere.r);
					leftBox.grow(sphere.pos + sphere.r);
				}
				else
				{
					rightCount++;
					rightBox.grow(sphere.pos - sphere.r);
					rightBox.grow(sphere.pos + sphere.r);
				}
			}
			else
			{
				Triangle& triangle = triangles[primitiveIdx[node.leftFirst + i] - spheres_size];
				if (triangle.centroid[axis] < pos)
				{
					leftCount++;
					leftBox.grow(triangle.pos1);
					leftBox.grow(triangle.pos2);
					leftBox.grow(triangle.pos3);
				}
				else
				{
					rightCount++;
					rightBox.grow(triangle.pos1);
					rightBox.grow(triangle.pos2);
					rightBox.grow(triangle.pos3);
				}
			}


		}
		float cost = leftCount * leftBox.area() + rightCount * rightBox.area();
		return cost > 0 ? cost : 1e30f;
	}

	bool IntersectAABB(const Ray& ray, const float3 bmin, const float3 bmax)
	{
		float tx1 = (bmin.x - ray.O.x) * ray.rD.x, tx2 = (bmax.x - ray.O.x) * ray.rD.x;
		float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
		float ty1 = (bmin.y - ray.O.y) * ray.rD.y, ty2 = (bmax.y - ray.O.y) * ray.rD.y;
		tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
		float tz1 = (bmin.z - ray.O.z) * ray.rD.z, tz2 = (bmax.z - ray.O.z) * ray.rD.z;
		tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
		return tmax >= tmin && tmin < ray.t&& tmax > 0;
	}

	float IntersectAABBStack(const Ray& ray, const float3 bmin, const float3 bmax)
	{
		float tx1 = (bmin.x - ray.O.x) * ray.rD.x, tx2 = (bmax.x - ray.O.x) * ray.rD.x;
		float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
		float ty1 = (bmin.y - ray.O.y) * ray.rD.y, ty2 = (bmax.y - ray.O.y) * ray.rD.y;
		tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
		float tz1 = (bmin.z - ray.O.z) * ray.rD.z, tz2 = (bmax.z - ray.O.z) * ray.rD.z;
		tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
		if (tmax >= tmin && tmin < ray.t && tmax > 0) return tmin; else return 1e30f;
	}

	__m128 IntersectAABBSIMD(const Ray& ray, const uint node, __m128 ray_t, __m128 ray_O_x, __m128 ray_rD_x, __m128 ray_O_y, __m128 ray_rD_y, __m128 ray_O_z, __m128 ray_rD_z)
	{

		__m128 tx1 = _mm_mul_ps(_mm_sub_ps(qbvhNode[node].bminx4, ray_O_x), ray_rD_x);
		__m128 tx2 = _mm_mul_ps(_mm_sub_ps(qbvhNode[node].bmaxx4, ray_O_x), ray_rD_x);

		__m128 tmin = _mm_min_ps(tx1, tx2);
		__m128 tmax = _mm_max_ps(tx1, tx2);

		__m128 ty1 = _mm_mul_ps(_mm_sub_ps(qbvhNode[node].bminy4, ray_O_y), ray_rD_y);
		__m128 ty2 = _mm_mul_ps(_mm_sub_ps(qbvhNode[node].bmaxy4, ray_O_y), ray_rD_y);

		tmin = _mm_max_ps(tmin, _mm_min_ps(ty1, ty2));
		tmax = _mm_min_ps(tmax, _mm_max_ps(ty1, ty2));

		__m128 tz1 = _mm_mul_ps(_mm_sub_ps(qbvhNode[node].bminz4, ray_O_z), ray_rD_z);
		__m128 tz2 = _mm_mul_ps(_mm_sub_ps(qbvhNode[node].bmaxz4, ray_O_z), ray_rD_z);

		tmin = _mm_max_ps(tmin, _mm_min_ps(tz1, tz2));
		tmax = _mm_min_ps(tmax, _mm_max_ps(tz1, tz2));

		return _mm_and_ps(_mm_cmpge_ps(tmax, tmin), _mm_and_ps(_mm_cmplt_ps(tmin, ray_t), _mm_cmpgt_ps(tmax, zero)));
		/*
		_mm_cmpge_ps(tmax, tmin);
		_mm_cmplt_ps(tmin, tmax);
		_mm_cmpgt_ps(tmax, zero);
		
		return tmax >= tmin && tmin < ray.t && tmax > 0;
		*/
	}

	void IntersectBVH(Ray& ray)
	{
		BVHNode& planeRootNode = bvhNode[planeRootNodeIdx];
		for (uint i = 0; i < planeRootNode.primitiveCount; i++)
		{
			planes[i].Intersect(ray);
		}

		IntersectBVH(ray, primitveRootNodeIdx);
	}

	void IntersectBVH(Ray& ray, const uint nodeIdx)
	{
		BVHNode& node = bvhNode[nodeIdx];
		if (!IntersectAABB(ray, node.aabbMin, node.aabbMax)) return;
		if (node.isLeaf())
		{
			for (uint i = 0; i < node.primitiveCount; i++)
			{
				if (primitiveIdx[node.leftFirst + i] < spheres_size)
				{
					spheres[primitiveIdx[node.leftFirst + i]].Intersect(ray);
				}
				else
				{
					triangles[primitiveIdx[node.leftFirst + i] - spheres_size].Intersect(ray);
				}
			}
		}
		else
		{
			IntersectBVH(ray, node.leftFirst);
			IntersectBVH(ray, node.leftFirst + 1);
		}
	}

	void IntersectBVHStack (Ray& ray)
	{
		BVHNode& planeRootNode = bvhNode[planeRootNodeIdx];
		for (uint i = 0; i < planeRootNode.primitiveCount; i++)
		{
			planes[i].Intersect(ray);
		}

		BVHNode* node = &bvhNode[primitveRootNodeIdx], * stack[64];
		uint stackPtr = 0;

		while (1)
		{
			if (node->isLeaf())
			{
				for (uint i = 0; i < node->primitiveCount; i++)
				{
					if (primitiveIdx[node->leftFirst + i] < spheres_size)
					{
						spheres[primitiveIdx[node->leftFirst + i]].Intersect(ray);
					}
					else
					{
						triangles[primitiveIdx[node->leftFirst + i] - spheres_size].Intersect(ray);
					}
				}

				if (stackPtr == 0) break; else node = stack[--stackPtr];
				continue;
			}

			BVHNode* child1 = &bvhNode[node->leftFirst];
			BVHNode* child2 = &bvhNode[node->leftFirst + 1];
			float dist1 = IntersectAABBStack(ray, child1->aabbMin, child1->aabbMax);
			float dist2 = IntersectAABBStack(ray, child2->aabbMin, child2->aabbMax);
			if (dist1 > dist2) { swap(dist1, dist2); swap(child1, child2); }
			if (dist1 == 1e30f)
			{
				if (stackPtr == 0) break; else node = stack[--stackPtr];
			}
			else
			{
				node = child1;
				if (dist2 != 1e30f) stack[stackPtr++] = child2;
			}
		}
	}
	void IntersectQBVH(Ray& ray)
	{
		SIMD_BVH_Node& rootNode = qbvhNode[0];

		// Plane child
		for (uint i = 0; i < rootNode.count[0]; i++)
		{
			planes[i].Intersect(ray);
		}

		// SIMD
		__m128 ray_t = _mm_set1_ps(ray.t);

		__m128 ray_O_x = _mm_set1_ps(ray.O.x);
		__m128 ray_rD_x = _mm_set1_ps(ray.rD.x);

		__m128 ray_O_y = _mm_set1_ps(ray.O.y);
		__m128 ray_rD_y = _mm_set1_ps(ray.rD.y);

		__m128 ray_O_z = _mm_set1_ps(ray.O.z);
		__m128 ray_rD_z = _mm_set1_ps(ray.rD.z);


		// Primitive Child
		IntersectQBVH(ray, rootNode.child[1], ray_t, ray_O_x, ray_rD_x, ray_O_y, ray_rD_y, ray_O_z, ray_rD_z);

	}

	void IntersectQBVH(Ray& ray, const uint nodeIdx, __m128 ray_t, __m128 ray_O_x, __m128 ray_rD_x, __m128 ray_O_y, __m128 ray_rD_y, __m128 ray_O_z, __m128 ray_rD_z)
	{
		SIMD_BVH_Node& parentNode = qbvhNode[nodeIdx];

		__m128 intersect = IntersectAABBSIMD(ray, nodeIdx, ray_t, ray_O_x, ray_rD_x, ray_O_y, ray_rD_y, ray_O_z, ray_rD_z);

		for (int i = 0; i < 4; i++)
		{
			if (parentNode.child[i] == -1) return;
			if (intersect.m128_i32[i] == 0) continue;


			if (parentNode.count[i] > 0)
			{
				for (int j = 0; j < parentNode.count[i]; j++)
				{
					if (primitiveIdx[parentNode.child[i] + j] < spheres_size)
					{
						spheres[primitiveIdx[parentNode.child[i] + j]].Intersect(ray);
					}
					else
					{
						triangles[primitiveIdx[parentNode.child[i] + j] - spheres_size].Intersect(ray);
					}
				}
			}
			else
			{
				IntersectQBVH(ray, parentNode.child[i], ray_t, ray_O_x, ray_rD_x, ray_O_y, ray_rD_y, ray_O_z, ray_rD_z);
			}
		}
	}

	bool IntersectBVHShadowRay(Ray& ray, bool isOccluded)
	{
		if (isOccluded)
			return isOccluded;

		BVHNode& planeRootNode = bvhNode[planeRootNodeIdx];
		for (uint i = 0; i < planeRootNode.primitiveCount; i++)
		{
			planes[i].Intersect(ray);
			if (ray.objIdx != -1)
				return true;
		}

		return IntersectBVHShadowRay(ray, isOccluded, primitveRootNodeIdx);
	}

	bool IntersectBVHShadowRay(Ray& ray, bool isOccluded, const uint nodeIdx)
	{
		if (isOccluded)
			return isOccluded;

		BVHNode& node = bvhNode[nodeIdx];

		if (!IntersectAABB(ray, node.aabbMin, node.aabbMax)) return isOccluded;
		if (node.isLeaf())
		{
			for (uint i = 0; i < node.primitiveCount; i++)
			{
				if (primitiveIdx[node.leftFirst + i] < spheres_size)
				{
					spheres[primitiveIdx[node.leftFirst + i]].Intersect(ray);
					if (ray.objIdx != -1)
						return true;
				}
				else
				{
					triangles[primitiveIdx[node.leftFirst + i] - spheres_size].Intersect(ray);
					if (ray.objIdx != -1)
						return true;
				}
			}
			return false;
		}
		else
		{
			bool leftOccluded = IntersectBVHShadowRay(ray, isOccluded, node.leftFirst);
			return IntersectBVHShadowRay(ray, leftOccluded, node.leftFirst + 1);
		}
	}

	bool IntersectQBVHShadowRay(Ray& ray, bool isOccluded)
	{
		if (isOccluded)
			return isOccluded;

		SIMD_BVH_Node& rootNode = qbvhNode[rootNodeIdx];
		for (uint i = 0; i < rootNode.count[0]; i++)
		{
			planes[i].Intersect(ray);
			if (ray.objIdx != -1)
				return true;
		}

		// SIMD
		__m128 ray_t = _mm_set1_ps(ray.t);

		__m128 ray_O_x = _mm_set1_ps(ray.O.x);
		__m128 ray_rD_x = _mm_set1_ps(ray.rD.x);

		__m128 ray_O_y = _mm_set1_ps(ray.O.y);
		__m128 ray_rD_y = _mm_set1_ps(ray.rD.y);

		__m128 ray_O_z = _mm_set1_ps(ray.O.z);
		__m128 ray_rD_z = _mm_set1_ps(ray.rD.z);

		return IntersectQBVHShadowRay(ray, isOccluded, rootNode.child[1], ray_t, ray_O_x, ray_rD_x, ray_O_y, ray_rD_y, ray_O_z, ray_rD_z);
	}

	bool IntersectQBVHShadowRay(Ray& ray, bool isOccluded, const uint nodeIdx, __m128 ray_t, __m128 ray_O_x, __m128 ray_rD_x, __m128 ray_O_y, __m128 ray_rD_y, __m128 ray_O_z, __m128 ray_rD_z)
	{
		if (isOccluded)
			return isOccluded;

		SIMD_BVH_Node& parentNode = qbvhNode[nodeIdx];
		__m128 intersect = IntersectAABBSIMD(ray, nodeIdx, ray_t, ray_O_x, ray_rD_x, ray_O_y, ray_rD_y, ray_O_z, ray_rD_z);


		bool prevOccluded = false;
		for (int j = 0; j < 4; j++)
		{
			if (parentNode.child[j] == -1) return false;
			if (intersect.m128_i32[j] == 0) continue;
			//if (!IntersectAABB(ray, float3(parentNode.bminx[j], parentNode.bminy[j], parentNode.bminz[j]), float3(parentNode.bmaxx[j], parentNode.bmaxy[j], parentNode.bmaxz[j]))) continue;

			if (parentNode.count[j] > 0)
			{
				for (uint i = 0; i < parentNode.count[j]; i++)
				{
					if (primitiveIdx[parentNode.child[j] + i] < spheres_size)
					{
						spheres[primitiveIdx[parentNode.child[j] + i]].Intersect(ray);
						if (ray.objIdx != -1)
							return true;
					}
					else
					{
						triangles[primitiveIdx[parentNode.child[j] + i] - spheres_size].Intersect(ray);
						if (ray.objIdx != -1)
							return true;
					}
				}
			}
			else
			{
				prevOccluded = prevOccluded || IntersectQBVHShadowRay(ray, prevOccluded, parentNode.child[j], ray_t, ray_O_x, ray_rD_x, ray_O_y, ray_rD_y, ray_O_z, ray_rD_z);
			}
		}
		return prevOccluded;
	}

};
