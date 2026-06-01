# Anh Dũng — Week 1 Deliverables

Bộ file này là phần việc tuần 1 của Anh Dũng trong đồ án Hệ thống phát hiện ngã Edge–Cloud.

## Files chính

```text
docs/architecture.md
docs/mqtt_schema.json
docs/api_contract.md
docs/repo_convention.md
docker-compose.skeleton.yml
```

## Nội dung đã chốt

- Kiến trúc Edge–Cloud–App.
- Flow dữ liệu: Camera → Jetson/DeepStream → MQTT → Cloud → FCM → Flutter app.
- Không dùng SMS trong main flow.
- Telegram không nằm trong main flow.
- Notification chính là FCM push notification tới Flutter app.
- Temporal logic dùng key `(camera_id, person_id)`.
- Fall detection không chỉ dựa vào class `lying`, mà dựa vào chuyển trạng thái `standing/sitting → lying` trong khoảng 2 giây.

## Cách gửi cho nhóm

1. Đưa cả folder `docs/` vào repo.
2. Gửi `mqtt_schema.json` cho Khánh để làm MQTT Consumer.
3. Gửi `api_contract.md` cho Khánh và Tấn Dũng để backend/app code cùng format.
4. Gửi `architecture.md` cho cả nhóm để thống nhất workflow.
5. Dùng `docker-compose.skeleton.yml` làm khung deploy ban đầu.