#include <cstdio>

int main()
{
	printf("Pre-increment results:\n");
	for (int i = 0; i < 5; ++i)
	{
		printf("i = %d\n", i);
	}

	printf("Post-increment results:\n");
	for (int i = 0; i < 5; i++)
	{
		printf("i = %d\n", i);
	}

	return 0;
}