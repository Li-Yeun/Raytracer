#include "primitive.h"
#pragma once

__declspec(align(32)) struct BVHNode
{
	float3 aabbMin, aabbMax;
	uint leftFirst, primitiveCount;
	bool isLeaf() { return primitiveCount > 0; }
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
		std::cout << nodesUsed << std::endl;
		std::cout << nodesUsedQBVH << std::endl;

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

						float area = areaX * areaY * areaZ;
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
		bvhNode = new BVHNode[(spheres_size + triangles_size) * 2 + 1]; // 1 leaf node reserved for planes
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
		if (node.primitiveCount <= 2) return;
		// determine split axis and position
		float3 extent = node.aabbMax - node.aabbMin;
		int axis = 0;
		if (extent.y > extent.x) axis = 1;
		if (extent.z > extent[axis]) axis = 2;
		float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;
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

	bool IntersectAABB(const Ray& ray, const float3 bmin, const float3 bmax)
	{
		float tx1 = (bmin.x - ray.O.x) / ray.D.x, tx2 = (bmax.x - ray.O.x) / ray.D.x;
		float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
		float ty1 = (bmin.y - ray.O.y) / ray.D.y, ty2 = (bmax.y - ray.O.y) / ray.D.y;
		tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
		float tz1 = (bmin.z - ray.O.z) / ray.D.z, tz2 = (bmax.z - ray.O.z) / ray.D.z;
		tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
		return tmax >= tmin && tmin < ray.t&& tmax > 0;
	}

	__m128 IntersectAABBSIMD(const Ray& ray, const uint node, __m128 ray_t, __m128 ray_O_x, __m128 ray_D_x, __m128 ray_O_y, __m128 ray_D_y, __m128 ray_O_z, __m128 ray_D_z)
	{
		__m128 tx1 = _mm_div_ps(_mm_sub_ps(qbvhNode[node].bminx4, ray_O_x), ray_D_x);
		__m128 tx2 = _mm_div_ps(_mm_sub_ps(qbvhNode[node].bmaxx4, ray_O_x), ray_D_x);

		//float tx1 = (bmin.x - ray.O.x) / ray.D.x, tx2 = (bmax.x - ray.O.x) / ray.D.x;

		__m128 tmin = _mm_min_ps(tx1, tx2);
		__m128 tmax = _mm_max_ps(tx1, tx2);

		//float tmin = min(tx1, tx2), tmax = max(tx1, tx2);

		__m128 ty1 = _mm_div_ps(_mm_sub_ps(qbvhNode[node].bminy4, ray_O_y), ray_D_y);
		__m128 ty2 = _mm_div_ps(_mm_sub_ps(qbvhNode[node].bmaxy4, ray_O_y), ray_D_y);

		//float ty1 = (bmin.y - ray.O.y) / ray.D.y, ty2 = (bmax.y - ray.O.y) / ray.D.y;

		tmin = _mm_max_ps(tmin, _mm_min_ps(ty1, ty2));
		tmax = _mm_min_ps(tmax, _mm_max_ps(ty1, ty2));

		//tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));

		__m128 tz1 = _mm_div_ps(_mm_sub_ps(qbvhNode[node].bminz4, ray_O_z), ray_D_z);
		__m128 tz2 = _mm_div_ps(_mm_sub_ps(qbvhNode[node].bmaxz4, ray_O_z), ray_D_z);

		//float tz1 = (bmin.z - ray.O.z) / ray.D.z, tz2 = (bmax.z - ray.O.z) / ray.D.z;

		tmin = _mm_max_ps(tmin, _mm_min_ps(tz1, tz2));
		tmax = _mm_min_ps(tmax, _mm_max_ps(tz1, tz2));

		//tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
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
		__m128 ray_D_x = _mm_set1_ps(ray.D.x);

		__m128 ray_O_y = _mm_set1_ps(ray.O.y);
		__m128 ray_D_y = _mm_set1_ps(ray.D.y);

		__m128 ray_O_z = _mm_set1_ps(ray.O.z);
		__m128 ray_D_z = _mm_set1_ps(ray.D.z);


		// Primitive Child
		IntersectQBVH(ray, rootNode.child[1], ray_t, ray_O_x, ray_D_x, ray_O_y, ray_D_y, ray_O_z, ray_D_z);

	}

	void IntersectQBVH(Ray& ray, const uint nodeIdx, __m128 ray_t, __m128 ray_O_x, __m128 ray_D_x, __m128 ray_O_y, __m128 ray_D_y, __m128 ray_O_z, __m128 ray_D_z)
	{
		SIMD_BVH_Node& parentNode = qbvhNode[nodeIdx];

		__m128 intersect = IntersectAABBSIMD(ray, nodeIdx, ray_t, ray_O_x, ray_D_x, ray_O_y, ray_D_y, ray_O_z, ray_D_z);

		for (int i = 0; i < 4; i++)
		{
			if (parentNode.child[i] == -1) return;
			if (intersect.m128_i32[i] == 0) continue;

			//if (!IntersectAABB(ray, float3(parentNode.bminx[i], parentNode.bminy[i], parentNode.bminz[i]), float3(parentNode.bmaxx[i], parentNode.bmaxy[i], parentNode.bmaxz[i]))) continue;

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
				IntersectQBVH(ray, parentNode.child[i], ray_t, ray_O_x, ray_D_x, ray_O_y, ray_D_y, ray_O_z, ray_D_z);
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
		__m128 ray_D_x = _mm_set1_ps(ray.D.x);

		__m128 ray_O_y = _mm_set1_ps(ray.O.y);
		__m128 ray_D_y = _mm_set1_ps(ray.D.y);

		__m128 ray_O_z = _mm_set1_ps(ray.O.z);
		__m128 ray_D_z = _mm_set1_ps(ray.D.z);

		return IntersectQBVHShadowRay(ray, isOccluded, rootNode.child[1], ray_t, ray_O_x, ray_D_x, ray_O_y, ray_D_y, ray_O_z, ray_D_z);
	}

	bool IntersectQBVHShadowRay(Ray& ray, bool isOccluded, const uint nodeIdx, __m128 ray_t, __m128 ray_O_x, __m128 ray_D_x, __m128 ray_O_y, __m128 ray_D_y, __m128 ray_O_z, __m128 ray_D_z)
	{
		if (isOccluded)
			return isOccluded;

		SIMD_BVH_Node& parentNode = qbvhNode[nodeIdx];
		__m128 intersect = IntersectAABBSIMD(ray, nodeIdx, ray_t, ray_O_x, ray_D_x, ray_O_y, ray_D_y, ray_O_z, ray_D_z);


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
				prevOccluded = prevOccluded || IntersectQBVHShadowRay(ray, prevOccluded, parentNode.child[j], ray_t, ray_O_x, ray_D_x, ray_O_y, ray_D_y, ray_O_z, ray_D_z);
			}
		}
		return prevOccluded;
	}

};

