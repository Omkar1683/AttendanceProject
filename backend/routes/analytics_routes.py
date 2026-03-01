"""
routes/analytics_routes.py
--------------------------
Blueprint: analytics summaries and downloadable reports.
URL prefix: /  (preserves original /analytics/* and /reports/* paths)
"""
import io
from datetime import datetime

from flask import Blueprint, request, jsonify, send_file

from core.security import token_required
import services.analytics_service as aly_svc

analytics_bp = Blueprint('analytics', __name__)


@analytics_bp.route('/analytics/today', methods=['GET'])
@token_required
def get_today_analytics():
    class_id = request.args.get('class_id')
    if not class_id:
        return jsonify({'status': 'error', 'message': 'Class ID required'}), 400
    summary = aly_svc.get_today_summary(class_id)
    return jsonify({'status': 'success', 'data': summary})


@analytics_bp.route('/analytics/defaulters', methods=['GET'])
@token_required
def get_defaulters():
    class_id  = request.args.get('class_id')
    threshold = int(request.args.get('threshold', 75))
    if not class_id:
        return jsonify({'status': 'error', 'message': 'Class ID required'}), 400
    defaulters = aly_svc.get_defaulters_list(class_id, threshold)
    return jsonify({'status': 'success', 'data': defaulters})


@analytics_bp.route('/reports/class', methods=['GET'])
@token_required
def get_class_report():
    class_id = request.args.get('class_id')
    month    = int(request.args.get('month', datetime.now().month))
    year     = int(request.args.get('year',  datetime.now().year))
    if not class_id:
        return jsonify({'status': 'error', 'message': 'Class ID required'}), 400
    report = aly_svc.get_monthly_report(class_id, month, year)
    return jsonify({'status': 'success', 'data': report})


@analytics_bp.route('/reports/student/<student_id>', methods=['GET'])
@token_required
def get_student_stats(student_id):
    report = aly_svc.get_student_report(student_id)
    return jsonify({'status': 'success', 'data': report})


@analytics_bp.route('/reports/export-csv', methods=['GET'])
@token_required
def export_report_csv():
    class_id = request.args.get('class_id')
    month    = int(request.args.get('month', datetime.now().month))
    year     = int(request.args.get('year',  datetime.now().year))
    report   = aly_svc.get_monthly_report(class_id, month, year)
    csv_data = aly_svc.export_to_csv(report, "Class Report")
    return send_file(
        io.BytesIO(csv_data.encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'attendance_report_{month}_{year}.csv',
    )
