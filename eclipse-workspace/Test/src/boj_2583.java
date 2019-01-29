import java.awt.Point;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Scanner;

public class boj_2583 {
	static int arr[][];
	static boolean visit[][];
	static int N, M, K;
	static int dx[] = { 0, 1, 0, -1 };
	static int dy[] = { 1, 0, -1, 0 };
	static int cnt = 0;
	static ArrayList<Integer> list;
	static Queue<Point> q;

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		list = new ArrayList<Integer>();
		N = sc.nextInt();
		// N = 5 M = 7
		M = sc.nextInt();
		K = sc.nextInt();
		arr = new int[M + 2][N + 2];
		visit = new boolean[M + 2][N + 2];
		q = new LinkedList<Point>();
		for (int i = 0; i < K; i++) {
			int x1 = sc.nextInt();
			int y1 = sc.nextInt();
			int x2 = sc.nextInt();
			int y2 = sc.nextInt();
			for (int a1 = x1; a1 <= (x2 - 1); a1++) {
				for (int b1 = y1; b1 <= (y2 - 1); b1++) {
					arr[a1][b1] = 1;
				}
			}
		}
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				if (arr[i][j] == 0 && !visit[i][j]) {
					q.add(new Point(i, j));
					bfs(i, j);
				}
			}
		}

		System.out.println(cnt);
		Collections.sort(list);
		for (int i = 0; i < list.size(); i++) {
			System.out.print(list.get(i)+" ");
		}

	}

	private static void bfs(int x, int y) {
		cnt++;
		int sum = 1;
		if(arr[x][y]==0)
		visit[x][y] = true;

		while (!q.isEmpty()) {
			Point temp = q.poll();
			
			for(int i=0; i<4; i++) {
				int nx = dx[i] + temp.x;
				int ny = dy[i] + temp.y;
				
				if(nx >= 0 && ny >=0 && nx < M && ny <N) {
					if(arr[nx][ny]==0 && !visit[nx][ny]) {
						visit[nx][ny] = true;
						sum++;
						q.add(new Point(nx, ny));
					}
				}
			}
		}
		if(sum!=0) {
			list.add(sum);
		}
	}

}
