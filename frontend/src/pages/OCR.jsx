import DashboardLayout from '../layouts/DashboardLayout';
import AnalysisPage from '../components/AnalysisPage';

export default function OCR() {
  return (
    <DashboardLayout>
      <AnalysisPage title="OCR & Price Detection" endpoint="/ocr/" />
    </DashboardLayout>
  );
}
