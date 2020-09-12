#include <bits/stdc++.h> 
using namespace std; 
void findMinimumOperations(double* a, int n) 
{ 
	long long ans = INT_MAX; 
	// The array c describes all the given set of 
	// possible operations. 
	int c[] = {-1,1}; 
	// Size of c 
	int possiblities = 2; 

	// candidate answer 
	int pos1 = -1, pos2 = -1; 

	// loop through all the permutations of the first two 
	// elements. 
	for (int i = 0; i < possiblities; i++) { 
		for (int j = 0; j < possiblities; j++) { 

			// a1 and a2 are the candidate first two elements 
			// of the possible GP. 
			double a1 = a[1] + c[i]; 
			double a2 = a[2] + c[j]; 

			// temp stores the current answer, including the 
			// modification of the first two elements. 
			int temp = abs(a1 - a[1]) + abs(a2 - a[2]); 

			if (a1 == 0 || a2 == 0) 
				continue; 

			// common ratio of the possible GP 
			double r = a2 / a1; 

			// To check if the chosen set is valid, and id yes 
			// find the number of operations it takes. 
			for (int pos = 3; pos <= n; pos++) { 

				// ai is value of a[i] according to the assumed 
				// first two elements a1, a2 
				// ith element of an GP = a1*((a2-a1)^(i-1)) 
				double ai = a1 * pow(r, pos - 1); 

				// Check for the "proposed" element to be only 
				// differing by one 
				if (a[pos] == ai) { 
					continue; 
				} 
				else if (a[pos] + 1 == ai || a[pos] - 1 == ai) { 
					temp++; 
				} 
				else { 
					temp = INT_MAX; // set the temporary ans 
					break; // to infinity and break 
				} 
			} 

			// update answer 
			if (temp < ans) { 
				ans = temp; 
				pos1 = a1; 
				pos2 = a2; 
			} 
		} 
	} 
	if (ans == -1) { 
		cout << "-1"; 
		return; 
	} 

	cout << "Minimum Number of Operations are " << ans << "\n";
} 
 
int main() 
{ 
    int n;
    cin>>n;
    int arr[n];
    for(int i = 0;i<n;i++) cin>>arr[i];
    sort(arr,arr+n);
    for(int i = 0;i<n;i++) cout<<arr[i];
	findMinimumOperations(arr, n - 1); 
	return 0; 
} 
