from django.http import FileResponse, HttpResponse
from django.shortcuts import render
import os
from django.conf import settings
import win32com.client as win32
import pythoncom
from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import os
import tempfile
from . import parse
def index(request):
    return render(request, 'main/index.html')

def download_hwp(request):
    print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
    pythoncom.CoInitialize()
    
    # result = parse.main(request)
    hwp = win32.gencache.EnsureDispatch("hwpframe.hwpobject")
    hwp.XHwpWindows.Item(0).Visible = True
    hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
    hwp_path = os.path.join(settings.BASE_DIR, 'main/static/final.hwpx')
    hwp.Open(hwp_path)
    
    hwp.SaveAs(hwp.Path.replace(".hwpx", "2.hwpx"))
    
    action    = hwp.HAction
    param_set = hwp.HParameterSet

    # InsertText용 파라미터 객체(HInsertText.HSet) 한 번만 받아두기
    insert_param = param_set.HInsertText

    # 1) 기관명 입력
    action.GetDefault("InsertText", param_set.HInsertText.HSet)
    insert_param.Text = "봄이"
    action.Execute("InsertText", param_set.HInsertText.HSet)

    # 2) 오른쪽 셀 3번 추가
    for _ in range(3):
        action.Run("TableRightCellAppend")

    # 3) 담당자명 입력
    action.GetDefault("InsertText", param_set.HInsertText.HSet)
    insert_param.Text = "이창환"
    action.Execute("InsertText", param_set.HInsertText.HSet)

    # 4) 오른쪽 셀 2번 추가
    for _ in range(2):
        action.Run("TableRightCellAppend")

    # 5) 연락처 입력
    action.GetDefault("InsertText", param_set.HInsertText.HSet)
    insert_param.Text = "010-1234-5678"
    action.Execute("InsertText", param_set.HInsertText.HSet)

    # 6) 오른쪽 셀 3번 추가
    for _ in range(3):
        action.Run("TableRightCellAppend")

    # 7) 기업명 입력
    action.GetDefault("InsertText", param_set.HInsertText.HSet)
    insert_param.Text = "한림대 연구실"
    action.Execute("InsertText", param_set.HInsertText.HSet)

    # 8) 오른쪽 셀 2번 추가
    for _ in range(2):
        action.Run("TableRightCellAppend")

    # 9) 사업자번호 입력
    action.GetDefault("InsertText", param_set.HInsertText.HSet)
    insert_param.Text = "333"
    action.Execute("InsertText", param_set.HInsertText.HSet)

    # 10) 오른쪽 셀 3번 추가
    for _ in range(3):
        action.Run("TableRightCellAppend")

    # 11) 대표자명 입력
    action.GetDefault("InsertText", param_set.HInsertText.HSet)
    insert_param.Text = "이민주"
    action.Execute("InsertText", param_set.HInsertText.HSet)

    # 12) 오른쪽 셀 2번 추가
    for _ in range(2):
        action.Run("TableRightCellAppend")

    # 13) 연락처1 입력
    action.GetDefault("InsertText", param_set.HInsertText.HSet)
    insert_param.Text = "010-1111-2222"
    action.Execute("InsertText", param_set.HInsertText.HSet)

    # 14) 아래로 이동
    action.Run("MoveDown")

    # 15) 연락처2 입력
    action.GetDefault("InsertText", param_set.HInsertText.HSet)
    insert_param.Text = "010-3333-4444"
    action.Execute("InsertText", param_set.HInsertText.HSet)

    # 16) 아래로 이동
    action.Run("MoveDown")

    # 17) 지원과제명 입력
    action.GetDefault("InsertText", param_set.HInsertText.HSet)
    insert_param.Text = "요약약"
    action.Execute("InsertText", param_set.HInsertText.HSet)

    # 18) 아래로 이동
    action.Run("MoveDown")

    # 19) 아이템 입력
    action.GetDefault("InsertText", param_set.HInsertText.HSet)
    insert_param.Text =  "ㅇㅇ"
    action.Execute("InsertText", param_set.HInsertText.HSet)

    # 20) 아래로 이동
    action.Run("MoveDown")

    # 21) 추천사유 입력
    action.GetDefault("InsertText", param_set.HInsertText.HSet)
    insert_param.Text = "ㄷㄷㄷ"
    action.Execute("InsertText", param_set.HInsertText.HSet)
    
    hwp.Save()
    hwp.Quit()

    
    
    
    hwp_path = os.path.join(settings.BASE_DIR, 'main/static/final2.hwpx')
    hwp.Open(hwp_path)
    return FileResponse(open(hwp_path, 'rb'), content_type='application/haansofthwp')




# myapp/views.py

def transmit_hwp(request):
    print("ASDSADASDASDASDASDAS")
    if request.method == 'POST':
        uploaded_file = request.FILES.get('hwp_file')
        if not uploaded_file:
            # 파일이 없으면 폼 렌더링하며 에러 메시지 출력
            return render(request, 'main/index.html', {
                'error': '파일을 선택해주세요.'
            })
        # 저장 경로 설정: MEDIA_ROOT/hwp_uploads/<파일명>
        save_dir = os.path.join(settings.MEDIA_ROOT, 'main/static')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "dummy.hwpx")
        # 파일 저장
        with open(save_path, 'wb+') as dest:
            for chunk in uploaded_file.chunks():
                dest.write(chunk)
        # 저장이 완료되면 빈 응답(204) 반환. 클라이언트에는 아무 콘텐츠도 보내지 않음.
        
        
        
        
        
        pythoncom.CoInitialize()
    
        result = parse.main(save_path)
        hwp = win32.gencache.EnsureDispatch("hwpframe.hwpobject")
        hwp.XHwpWindows.Item(0).Visible = True
        hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
        hwp_path = os.path.join(settings.BASE_DIR, 'main/static/final.hwpx')
        hwp.Open(hwp_path)
        
        hwp.SaveAs(hwp.Path.replace(".hwpx", "2.hwpx"))
        
        action    = hwp.HAction
        param_set = hwp.HParameterSet

        # InsertText용 파라미터 객체(HInsertText.HSet) 한 번만 받아두기
        insert_param = param_set.HInsertText

        # 1) 기관명 입력
        action.GetDefault("InsertText", param_set.HInsertText.HSet)
        insert_param.Text = result.get("기관명")
        action.Execute("InsertText", param_set.HInsertText.HSet)

        # 2) 오른쪽 셀 3번 추가
        for _ in range(3):
            action.Run("TableRightCellAppend")

        # 3) 담당자명 입력
        action.GetDefault("InsertText", param_set.HInsertText.HSet)
        insert_param.Text = result.get("담당자명")
        action.Execute("InsertText", param_set.HInsertText.HSet)

        # 4) 오른쪽 셀 2번 추가
        for _ in range(2):
            action.Run("TableRightCellAppend")

        # 5) 연락처 입력
        action.GetDefault("InsertText", param_set.HInsertText.HSet)
        insert_param.Text = result.get("연락처")
        action.Execute("InsertText", param_set.HInsertText.HSet)

        # 6) 오른쪽 셀 3번 추가
        for _ in range(3):
            action.Run("TableRightCellAppend")

        # 7) 기업명 입력
        action.GetDefault("InsertText", param_set.HInsertText.HSet)
        insert_param.Text = result.get("기업명")
        action.Execute("InsertText", param_set.HInsertText.HSet)

        # 8) 오른쪽 셀 2번 추가
        for _ in range(2):
            action.Run("TableRightCellAppend")

        # 9) 사업자번호 입력
        action.GetDefault("InsertText", param_set.HInsertText.HSet)
        insert_param.Text = result.get("사업자번호")
        action.Execute("InsertText", param_set.HInsertText.HSet)

        # 10) 오른쪽 셀 3번 추가
        for _ in range(3):
            action.Run("TableRightCellAppend")

        # 11) 대표자명 입력
        action.GetDefault("InsertText", param_set.HInsertText.HSet)
        insert_param.Text = result.get("대표자명")
        action.Execute("InsertText", param_set.HInsertText.HSet)

        # 12) 오른쪽 셀 2번 추가
        for _ in range(2):
            action.Run("TableRightCellAppend")

        # 13) 연락처1 입력
        action.GetDefault("InsertText", param_set.HInsertText.HSet)
        insert_param.Text = result.get("연락처1")
        action.Execute("InsertText", param_set.HInsertText.HSet)

        # 14) 아래로 이동
        action.Run("MoveDown")

        # 15) 연락처2 입력
        action.GetDefault("InsertText", param_set.HInsertText.HSet)
        insert_param.Text = result.get("연락처2")
        action.Execute("InsertText", param_set.HInsertText.HSet)

        # 16) 아래로 이동
        action.Run("MoveDown")

        # 17) 지원과제명 입력
        action.GetDefault("InsertText", param_set.HInsertText.HSet)
        insert_param.Text = result.get("지원과제명")
        action.Execute("InsertText", param_set.HInsertText.HSet)

        # 18) 아래로 이동
        action.Run("MoveDown")

        # 19) 아이템 입력
        action.GetDefault("InsertText", param_set.HInsertText.HSet)
        insert_param.Text =  result.get("아이템")
        action.Execute("InsertText", param_set.HInsertText.HSet)

        # 20) 아래로 이동
        action.Run("MoveDown")

        # 21) 추천사유 입력
        action.GetDefault("InsertText", param_set.HInsertText.HSet)
        insert_param.Text = result.get("추천사유")
        action.Execute("InsertText", param_set.HInsertText.HSet)
        
        hwp.Save()
        hwp.Quit()

        
        
        
        hwp_path = os.path.join(settings.BASE_DIR, 'main/static/final2.hwpx')
        hwp.Open(hwp_path)
        return FileResponse(open(hwp_path, 'rb'), content_type='application/haansofthwp')
    # GET 요청: 업로드 폼 렌더링
    return render(request, 'main/index.html')








