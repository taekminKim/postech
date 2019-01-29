import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Scanner;
import java.util.StringTokenizer;

public class boj_1012 {
	static int arr[][];
	static int visit[][];
	static int dx[] = { 0, 1, 0, -1 };
	static int dy[] = { 1, 0, -1, 0 };
	static int cnt;
	static int N, M, K;

	public static void main(String[] args) throws NumberFormatException, IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		Scanner sc = new Scanner(System.in);
		int TC = Integer.parseInt(br.readLine());

		for (int test_case = 1; test_case <= TC; test_case++) {
			cnt = 0;
			StringTokenizer st = new StringTokenizer(br.readLine());
			N = Integer.parseInt(st.nextToken());
			M = Integer.parseInt(st.nextToken());
			K = Integer.parseInt(st.nextToken());
			arr = new int[N][M];
			visit = new int[N][M];

			for (int i = 0; i < K; i++) {
				st = new StringTokenizer(br.readLine());
				int x = Integer.parseInt(st.nextToken());
				int y = Integer.parseInt(st.nextToken());
				arr[x][y] = 1;
			}

			for (int i = 0; i < N; i++) {
				for (int j = 0; j < M; j++) {
					if (arr[i][j] == 1 && visit[i][j]==0) {
						bfs(i, j);
					}
				}
			}
			System.out.println(cnt);
		}
	}

	private static void bfs(int x, int y) {
		visit[x][y] = 1;

		Queue<Point> q = new LinkedList<Point>();
		q.add(new Point(x, y));
		while (!q.isEmpty()) {
			Point temp_p = q.poll();
			for (int i = 0; i < 4; i++) {
				int nx = temp_p.x + dx[i];
				int ny = temp_p.y + dy[i];

				if (nx >= 0 && ny >= 0 && nx < N && ny < M) {
					if (visit[nx][ny]==0 && arr[nx][ny]==1) {
						q.add(new Point(nx, ny));
						visit[nx][ny] = 1;
					}
				}
			}
		}
		cnt++;
	}
}

class Point {
	int x;
	int y;

	public Point(int x, int y) {
		super();
		this.x = x;
		this.y = y;
	}

}
