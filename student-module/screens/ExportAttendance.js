/**
 * student-module/screens/ExportAttendance.js
 * ---------------------------------------------
 * Feature 9: Export attendance as CSV or PDF.
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View, Text, ScrollView, StyleSheet, TouchableOpacity,
  Alert, ActivityIndicator, RefreshControl,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors, FontSizes, Spacing, BorderRadius } from '../constants/theme';
import { DashboardSkeleton } from '../components/LoadingSkeleton';
import ErrorState from '../components/ErrorState';
import ProgressBar from '../components/ProgressBar';
import studentApi from '../services/studentApi';

const ExportAttendance = ({ navigateTo }) => {
  const [report, setReport] = useState(null);
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [exportingCSV, setExportingCSV] = useState(false);
  const [exportingPDF, setExportingPDF] = useState(false);

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const [reportRes, profileRes] = await Promise.all([
        studentApi.getReport(),
        studentApi.getProfile(),
      ]);
      if (reportRes.status === 'success') setReport(reportRes.data);
      if (profileRes.status === 'success') setProfile(profileRes.data);
    } catch (err) {
      setError(err.message || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const handleExportCSV = async () => {
    setExportingCSV(true);
    try {
      const filePath = await studentApi.exportCSV(profile?.name || 'student');
      Alert.alert('Success', 'CSV file has been shared/saved.');
    } catch (err) {
      Alert.alert('Export Failed', err.message || 'Could not export CSV');
    } finally {
      setExportingCSV(false);
    }
  };

  const handleExportPDF = async () => {
    setExportingPDF(true);
    try {
      const filePath = await studentApi.exportPDF(profile?.name || 'student');
      Alert.alert('Success', 'PDF report has been shared/saved.');
    } catch (err) {
      Alert.alert('Export Failed', err.message || 'Could not export PDF');
    } finally {
      setExportingPDF(false);
    }
  };

  if (loading) return <DashboardSkeleton />;
  if (error) return <ErrorState message={error} onRetry={loadData} />;

  const subjects = report?.subjects || [];
  const overallPct = report?.overall_percentage || 0;

  return (
    <ScrollView
      style={expStyles.container}
      contentContainerStyle={expStyles.content}
      showsVerticalScrollIndicator={false}
    >
      {/* Preview Card */}
      <View style={expStyles.card}>
        <Text style={expStyles.cardLabel}>EXPORT PREVIEW</Text>
        <View style={expStyles.previewHeader}>
          <Text style={expStyles.previewName}>{profile?.name || 'Student'}</Text>
          <Text style={expStyles.previewDetail}>
            {profile?.roll_no && `${profile.roll_no} • `}
            {profile?.department || ''}
          </Text>
        </View>

        <View style={expStyles.overallRow}>
          <Text style={expStyles.overallLabel}>Overall Attendance</Text>
          <Text style={[expStyles.overallPct, {
            color: overallPct >= 75 ? Colors.success : Colors.danger,
          }]}>
            {overallPct}%
          </Text>
        </View>
        <ProgressBar percentage={overallPct} height={8} />

        {/* Summary Table */}
        <View style={expStyles.table}>
          <View style={expStyles.tableHeader}>
            <Text style={[expStyles.tableCell, expStyles.headerText, { flex: 2 }]}>Subject</Text>
            <Text style={[expStyles.tableCell, expStyles.headerText]}>Present</Text>
            <Text style={[expStyles.tableCell, expStyles.headerText]}>Total</Text>
            <Text style={[expStyles.tableCell, expStyles.headerText]}>%</Text>
          </View>
          {subjects.map((subj, i) => (
            <View key={i} style={expStyles.tableRow}>
              <Text style={[expStyles.tableCell, { flex: 2 }]} numberOfLines={1}>{subj.name}</Text>
              <Text style={expStyles.tableCell}>{subj.present}</Text>
              <Text style={expStyles.tableCell}>{subj.total_classes}</Text>
              <Text style={[expStyles.tableCell, {
                color: subj.percentage >= 75 ? Colors.success : Colors.danger,
                fontWeight: '600',
              }]}>
                {subj.percentage}%
              </Text>
            </View>
          ))}
        </View>
      </View>

      {/* Export Buttons */}
      <View style={expStyles.card}>
        <Text style={expStyles.cardLabel}>EXPORT OPTIONS</Text>

        <TouchableOpacity
          style={[expStyles.exportBtn, { borderColor: Colors.success }]}
          onPress={handleExportCSV}
          disabled={exportingCSV}
          activeOpacity={0.7}
        >
          <View style={[expStyles.exportIcon, { backgroundColor: Colors.successLight }]}>
            {exportingCSV ? (
              <ActivityIndicator size="small" color={Colors.success} />
            ) : (
              <Ionicons name="document-text-outline" size={24} color={Colors.success} />
            )}
          </View>
          <View style={expStyles.exportInfo}>
            <Text style={expStyles.exportTitle}>Export as CSV</Text>
            <Text style={expStyles.exportDesc}>
              Spreadsheet format — open in Excel, Google Sheets
            </Text>
          </View>
          <Ionicons name="share-outline" size={20} color={Colors.textMuted} />
        </TouchableOpacity>

        <TouchableOpacity
          style={[expStyles.exportBtn, { borderColor: Colors.danger }]}
          onPress={handleExportPDF}
          disabled={exportingPDF}
          activeOpacity={0.7}
        >
          <View style={[expStyles.exportIcon, { backgroundColor: Colors.dangerLight }]}>
            {exportingPDF ? (
              <ActivityIndicator size="small" color={Colors.danger} />
            ) : (
              <Ionicons name="document-outline" size={24} color={Colors.danger} />
            )}
          </View>
          <View style={expStyles.exportInfo}>
            <Text style={expStyles.exportTitle}>Export as PDF</Text>
            <Text style={expStyles.exportDesc}>
              Formatted report — print or email to parents
            </Text>
          </View>
          <Ionicons name="share-outline" size={20} color={Colors.textMuted} />
        </TouchableOpacity>
      </View>

      {/* Info */}
      <View style={expStyles.infoCard}>
        <Ionicons name="information-circle-outline" size={16} color={Colors.primary} />
        <Text style={expStyles.infoText}>
          Reports include all your attendance data across all subjects. The export will open your
          device's share dialog to save or send the file.
        </Text>
      </View>

      <View style={{ height: 24 }} />
    </ScrollView>
  );
};

const expStyles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  content: { padding: Spacing.lg },
  card: {
    backgroundColor: Colors.surface, borderRadius: BorderRadius.xl,
    padding: Spacing.lg, marginBottom: Spacing.lg,
    borderWidth: 1, borderColor: Colors.border,
  },
  cardLabel: {
    fontSize: FontSizes.xs, fontWeight: 'bold', color: Colors.textMuted,
    marginBottom: Spacing.md, letterSpacing: 1,
  },
  // Preview
  previewHeader: { marginBottom: Spacing.md },
  previewName: { fontSize: FontSizes.xl, fontWeight: 'bold', color: Colors.text },
  previewDetail: { fontSize: FontSizes.sm, color: Colors.textTertiary, marginTop: 2 },
  overallRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    marginBottom: Spacing.sm,
  },
  overallLabel: { fontSize: FontSizes.sm, color: Colors.textSecondary },
  overallPct: { fontSize: FontSizes.xl, fontWeight: 'bold' },
  // Table
  table: { marginTop: Spacing.lg },
  tableHeader: {
    flexDirection: 'row', borderBottomWidth: 2, borderBottomColor: Colors.border,
    paddingBottom: Spacing.sm,
  },
  tableRow: {
    flexDirection: 'row', borderBottomWidth: 1, borderBottomColor: Colors.borderLight,
    paddingVertical: Spacing.sm,
  },
  tableCell: { flex: 1, fontSize: FontSizes.sm, color: Colors.text },
  headerText: { fontWeight: 'bold', color: Colors.textTertiary, fontSize: FontSizes.xs },
  // Export Buttons
  exportBtn: {
    flexDirection: 'row', alignItems: 'center',
    padding: Spacing.lg, borderRadius: BorderRadius.lg,
    borderWidth: 1, marginBottom: Spacing.sm,
  },
  exportIcon: {
    width: 48, height: 48, borderRadius: 12,
    alignItems: 'center', justifyContent: 'center', marginRight: Spacing.md,
  },
  exportInfo: { flex: 1 },
  exportTitle: { fontSize: FontSizes.md, fontWeight: '600', color: Colors.text },
  exportDesc: { fontSize: FontSizes.xs, color: Colors.textMuted, marginTop: 2 },
  // Info
  infoCard: {
    flexDirection: 'row', backgroundColor: Colors.primaryLight,
    borderRadius: BorderRadius.lg, padding: Spacing.md,
  },
  infoText: {
    fontSize: FontSizes.xs, color: Colors.primary,
    marginLeft: Spacing.sm, flex: 1, lineHeight: 18,
  },
});

export default ExportAttendance;
