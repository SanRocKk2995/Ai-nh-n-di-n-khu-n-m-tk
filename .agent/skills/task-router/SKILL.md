---
name: task-router
description: Route coding tasks so Antigravity orchestrates, Codex CLI implements, and Copilot CLI reviews.
---

# Task Router Skill

## Mục tiêu
Skill này dùng để điều phối workflow:

1. đọc task
2. giao Codex CLI làm phần code
3. giao Copilot CLI review phần thay đổi
4. lưu log vào thư mục `logs/`
5. trả kết quả tóm tắt để agent điều phối quyết định bước tiếp theo

## Khi nào dùng
Dùng skill này khi:
- cần giao việc code cho Codex CLI
- cần giao việc review cho Copilot CLI
- muốn workflow có log rõ ràng
- muốn tránh để nhiều agent cùng sửa file cùng lúc

## Nguyên tắc
- Antigravity chỉ điều phối, không sửa bừa khi đã giao worker
- Codex CLI chịu trách nhiệm implement
- Copilot CLI chịu trách nhiệm review
- Không để Codex và Copilot sửa cùng file trong cùng một lượt
- Kết quả mỗi bước phải được lưu vào `logs/`

## Cấu trúc liên quan
- `prompts/plan.txt`
- `prompts/implement.txt`
- `prompts/review.txt`
- `logs/codex.out`
- `logs/copilot.out`
- `logs/orchestrate.out`
- `logs/result.json`

## Workflow chuẩn

### Bước 1
Đọc prompt implement từ:
- `prompts/implement.txt`

### Bước 2
Gọi script:
- `scripts/run_codex.sh`

Mục tiêu:
- để Codex CLI đọc yêu cầu
- implement thay đổi
- ghi output vào `logs/codex.out`

### Bước 3
Gọi script:
- `scripts/run_copilot.sh`

Mục tiêu:
- để Copilot CLI review thay đổi hiện tại
- ghi output vào `logs/copilot.out`

### Bước 4
Gọi script:
- `scripts/orchestrate.sh`

Script này sẽ:
- tạo thư mục logs nếu chưa có
- chạy Codex
- chạy Copilot
- tạo `logs/result.json`

## Đầu ra mong muốn
Sau mỗi lần chạy, skill phải tạo được:
- `logs/codex.out`
- `logs/copilot.out`
- `logs/orchestrate.out`
- `logs/result.json`

## Format `result.json`
Ví dụ:

```json
{
  "status": "ok",
  "workflow": "antigravity -> codex -> copilot",
  "steps": [
    {
      "name": "implement",
      "agent": "codex",
      "log": "logs/codex.out"
    },
    {
      "name": "review",
      "agent": "copilot",
      "log": "logs/copilot.out"
    }
  ]
}