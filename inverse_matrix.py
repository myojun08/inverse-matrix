"""
역행렬 계산 프로그램 (이산수학 과제)

- 행렬식(여인수 전개 + 수반행렬)을 이용한 역행렬 계산
- 가우스-조던 소거법을 이용한 역행렬 계산
- 두 방법의 결과 출력 및 비교

모든 계산은 fractions.Fraction(유리수)으로 수행하여 부동소수점 오차가 없다.
"""
from fractions import Fraction


class SingularMatrixError(Exception):
    """역행렬이 존재하지 않을 때(특이행렬) 발생시키는 예외"""


# ============================================================
# 공통: 출력 도우미
# ============================================================
OUTPUT_MODE = "fraction"  # "fraction"(분수) 또는 "decimal"(소수)


def fmt(x):
    """원소 하나를 현재 출력 형식에 맞는 문자열로 변환"""
    if OUTPUT_MODE == "decimal":
        return f"{float(x):.4f}"
    return str(x)  # Fraction -> '3', '-7/10' 형태


def print_matrix(M, title=None, split=None):
    """행렬을 열 너비를 맞춰 출력. split이 주어지면 그 열 앞에 '|' 구분선(첨가행렬)"""
    if title:
        print(f"  {title}")
    cells = [[fmt(x) for x in row] for row in M]
    width = max(len(c) for row in cells for c in row)
    for row in cells:
        parts = [c.rjust(width) for c in row]
        if split is not None:
            line = "  ".join(parts[:split]) + "  |  " + "  ".join(parts[split:])
        else:
            line = "  ".join(parts)
        print(f"    [ {line} ]")


def print_header(tag, title):
    print()
    print("=" * 48)
    print(f" [{tag}] {title}")
    print("=" * 48)


# ============================================================
# 구성 조건 1: 행렬 입력
# ============================================================
def read_size():
    """행렬의 크기 n(1 이상의 정수)을 입력받는다. 잘못된 값이면 다시 입력받는다."""
    while True:
        s = input("행렬의 크기 n을 입력하세요 (n x n): ").strip()
        try:
            n = int(s)
        except ValueError:
            print("  [입력 오류] 정수를 입력하세요.")
            continue
        if n < 1:
            print("  [입력 오류] n은 1 이상이어야 합니다.")
            continue
        return n


def read_matrix(n):
    """n x n 정방행렬을 행 단위로 입력받아 2차원 리스트로 반환한다."""
    print(f"{n} x {n} 행렬을 한 행씩 입력하세요 (원소는 공백으로 구분).")
    matrix = []
    while len(matrix) < n:
        tokens = input(f"  {len(matrix) + 1}행: ").split()
        if len(tokens) != n:
            print(f"  [입력 오류] {n}개의 원소가 필요합니다. (입력한 개수: {len(tokens)}) 다시 입력하세요.")
            continue
        try:
            row = [Fraction(t) for t in tokens]  # 정수, 소수(0.5), 분수(1/2) 모두 허용
        except (ValueError, ZeroDivisionError):
            print("  [입력 오류] 숫자가 아닌 값이 있습니다. 다시 입력하세요.")
            continue
        matrix.append(row)
    return matrix


# ============================================================
# 구성 조건 2: 행렬식을 이용한 역행렬
#   A^-1 = adj(A) / det(A),  adj(A) = (여인수행렬)^T
# ============================================================
def minor(M, row, col):
    """row행과 col열을 제거한 소행렬"""
    return [r[:col] + r[col + 1:] for i, r in enumerate(M) if i != row]


def determinant(M):
    """여인수 전개(1행 기준)로 행렬식을 재귀적으로 계산"""
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    det = Fraction(0)
    for j in range(n):
        if M[0][j] != 0:
            det += (-1) ** j * M[0][j] * determinant(minor(M, 0, j))
    return det


def cofactor_matrix(M):
    """각 원소의 여인수 C_ij = (-1)^(i+j) * M_ij 로 이루어진 여인수행렬"""
    n = len(M)
    if n == 1:
        return [[Fraction(1)]]
    return [[(-1) ** (i + j) * determinant(minor(M, i, j)) for j in range(n)]
            for i in range(n)]


def transpose(M):
    return [list(row) for row in zip(*M)]


def inverse_by_determinant(M):
    """행렬식을 이용해 역행렬을 계산한다.
    반환값: (행렬식, 여인수행렬, 수반행렬, 역행렬)
    행렬식이 0이면 SingularMatrixError를 발생시킨다."""
    det = determinant(M)
    if det == 0:
        raise SingularMatrixError("행렬식이 0이므로 역행렬이 존재하지 않습니다.")
    cof = cofactor_matrix(M)
    adj = transpose(cof)
    inv = [[x / det for x in row] for row in adj]
    return det, cof, adj, inv


# ============================================================
# 구성 조건 3: 가우스-조던 소거법을 이용한 역행렬
#   [A | I]  --(기본 행 연산)-->  [I | A^-1]
# ============================================================
def inverse_by_gauss_jordan(M, steps=None):
    """가우스-조던 소거법으로 역행렬을 계산한다.
    steps에 리스트를 넘기면 (행 연산 설명, 첨가행렬 상태)를 차례로 기록한다.
    피벗을 찾을 수 없으면 SingularMatrixError를 발생시킨다."""
    n = len(M)
    aug = [list(M[i]) + [Fraction(int(i == j)) for j in range(n)] for i in range(n)]

    def log(desc):
        if steps is not None:
            steps.append((desc, [row[:] for row in aug]))

    log("초기 첨가행렬 [A | I]")
    for col in range(n):
        # (1) 피벗 선택: col열에서 0이 아닌 원소가 있는 행을 찾는다
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            raise SingularMatrixError(
                f"{col + 1}열에 0이 아닌 피벗이 없어 단위행렬로 만들 수 없습니다. "
                "역행렬이 존재하지 않습니다.")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
            log(f"R{col + 1} <-> R{pivot + 1}  (행 교환)")

        # (2) 피벗을 1로 만든다
        p = aug[col][col]
        if p != 1:
            aug[col] = [x / p for x in aug[col]]
            log(f"R{col + 1} <- R{col + 1} / ({fmt(p)})")

        # (3) 같은 열의 다른 행을 모두 0으로 만든다
        for r in range(n):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[col])]
                log(f"R{r + 1} <- R{r + 1} - ({fmt(f)}) x R{col + 1}")

    return [row[n:] for row in aug]


# ============================================================
# 구성 조건 4: 결과 비교
# ============================================================
def compare_matrices(A, B, tol=Fraction(0)):
    """두 행렬의 원소별 최대 차이를 구해 tol 이하이면 동일하다고 판단"""
    max_diff = max(abs(a - b) for ra, rb in zip(A, B) for a, b in zip(ra, rb))
    return max_diff <= tol, max_diff


# ============================================================
# 추가 기능: 검증 (A x A^-1 = I)
# ============================================================
def multiply(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
            for i in range(len(A))]


def is_identity(M):
    return all(M[i][j] == (1 if i == j else 0) for i in range(len(M)) for j in range(len(M)))


def ask_yes_no(prompt):
    while True:
        ans = input(prompt).strip().lower()
        if ans in ("y", "n"):
            return ans == "y"
        print("  [입력 오류] y 또는 n을 입력하세요.")


# ============================================================
# 메인 흐름
# ============================================================
def run_once():
    # ---------- 1. 행렬 입력 ----------
    print_header("1", "행렬 입력")
    n = read_size()
    A = read_matrix(n)
    print()
    print_matrix(A, "입력된 행렬 A =")

    # ---------- 2. 행렬식을 이용한 역행렬 ----------
    print_header("2", "행렬식을 이용한 역행렬 계산")
    inv_det = None
    try:
        det, cof, adj, inv_det = inverse_by_determinant(A)
        print(f"  det(A) = {fmt(det)}")
        print_matrix(cof, "여인수행렬 C =")
        print_matrix(adj, "수반행렬 adj(A) = C^T =")
        print_matrix(inv_det, "A^-1 = adj(A) / det(A) =")
    except SingularMatrixError as e:
        print("  det(A) = 0")
        print(f"  [오류] {e}")

    # ---------- 3. 가우스-조던 소거법 ----------
    print_header("3", "가우스-조던 소거법을 이용한 역행렬 계산")
    show_steps = ask_yes_no("소거 과정을 단계별로 출력할까요? (y/n): ")
    steps = []
    inv_gj, gj_error = None, None
    try:
        inv_gj = inverse_by_gauss_jordan(A, steps)
    except SingularMatrixError as e:
        gj_error = e
    if show_steps:
        for k, (desc, state) in enumerate(steps):
            print(f"  ({k}) {desc}")
            print_matrix(state, split=n)
    if inv_gj is not None:
        print_matrix(inv_gj, "A^-1 (가우스-조던) =")
    else:
        print(f"  [오류] {gj_error}")

    # ---------- 4. 결과 출력 및 비교 ----------
    print_header("4", "결과 출력 및 비교")
    if inv_det is None and inv_gj is None:
        print("  두 방법 모두 '역행렬이 존재하지 않음'으로 판정했습니다.")
        print("  => 결과 일치 (특이행렬)")
    elif inv_det is None or inv_gj is None:
        print("  한 방법만 역행렬을 계산했습니다. => 결과 불일치")
    else:
        print_matrix(inv_det, "[행렬식 방법]")
        print_matrix(inv_gj, "[가우스-조던 방법]")
        same, max_diff = compare_matrices(inv_det, inv_gj)
        print(f"  원소별 최대 차이: {fmt(max_diff)}")
        if same:
            print("  => 두 방법으로 구한 역행렬이 동일합니다.")
        else:
            print("  => 두 방법으로 구한 역행렬이 서로 다릅니다.")

    # ---------- 추가: 검증 ----------
    if inv_gj is not None:
        print_header("+", "검증: A x A^-1 = I ?")
        P = multiply(A, inv_gj)
        print_matrix(P, "A x A^-1 =")
        print("  => 단위행렬입니다. 역행렬 계산이 올바릅니다." if is_identity(P)
              else "  => 단위행렬이 아닙니다!")


def main():
    global OUTPUT_MODE
    print("*" * 48)
    print("   역행렬 계산 프로그램 (행렬식 / 가우스-조던)")
    print("*" * 48)
    while True:
        choice = input("출력 형식을 선택하세요 (1: 분수, 2: 소수): ").strip()
        if choice in ("1", "2"):
            OUTPUT_MODE = "fraction" if choice == "1" else "decimal"
            break
        print("  [입력 오류] 1 또는 2를 입력하세요.")

    run_once()


if __name__ == "__main__":
    main()
