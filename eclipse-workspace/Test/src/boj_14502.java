import java.awt.Point;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.Queue;

public class boj_14502 {
	static int arr[][];
	static int temp[][];
	static int dx[] = { 0, 1, 0, -1 };
	static int dy[] = { 1, 0, -1, 0 };
	static int N, M;

	public static void main(String[] args) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		String str[] = br.readLine().split(" ");
		N = Integer.parseInt(str[0]);
		M = Integer.parseInt(str[1]);
		arr = new int[N][M];
		temp = new int[N][M];
		for (int i = 0; i < N; i++) {
			str = br.readLine().split(" ");
			for (int j = 0; j < M; j++) {
				arr[i][j] = Integer.parseInt(str[j]);
			}
		}
		Reset();
		int max = Integer.MIN_VALUE;
		// 벽 설치
		for (int x1 = 0; x1 < N; x1++) {
			for (int y1 = 0; y1 < M; y1++) {
				if (temp[x1][y1] != 0)
					continue;
				for (int x2 = 0; x2 < N; x2++) {
					for (int y2 = 0; y2 < M; y2++) {
						if (x1 == x2 && y1 == y2)
							continue;
						if (temp[x2][y2] != 0)
							continue;
						for (int x3 = 0; x3 < N; x3++) {
							for (int y3 = 0; y3 < M; y3++) {
								if (x3 == x1 && y3 == y1)
									continue;
								if (x3 == x2 && y3 == y2)
									continue;
								if (temp[x3][y3] != 0)
									continue;
								for (int i = 0; i < N; i++) {
									for (int j = 0; j < M; j++) {
										if (temp[i][j] == 2) {
											// 바이러스일 경우에 실행
											temp[x1][y1] = 1;
											temp[x2][y2] = 1;
											temp[x3][y3] = 1;
											bfs(i, j);
										}
									}
								}
								max = Math.max(countArea(), max);
								Reset();
							}
						}
					}
				}
			}
		}
		System.out.println(max);
	}

	public static void bfs(int x, int y) {
		Queue<Point> q = new LinkedList<>();
		q.add(new Point(x, y));
		
		while (!q.isEmpty()) {
			Point d = q.poll();
			for (int i = 0; i < 4; i++) {
				int nx = dx[i] + d.x;
				int ny = dy[i] + d.y;
				if (nx >= 0 && ny >= 0 && nx < N && ny < M) {
					if (temp[nx][ny] == 0) {
						temp[nx][ny] = 2;
						q.add(new Point(nx, ny));
					}
				}
			}
		}
	}

	public static void Reset() {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				temp[i][j] = arr[i][j];
			}
		}
	}

	public static int countArea() {
		int cnt = 0;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				if (temp[i][j] == 0)
					cnt++;
			}
		}
		return cnt;
	}
}
