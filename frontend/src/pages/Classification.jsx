import DashboardLayout from '../layouts/DashboardLayout';
import AnalysisPage from '../components/AnalysisPage';

export default function Classification() {
  return (
    <DashboardLayout>
      <AnalysisPage title="Product Classification" endpoint="/classify-products/" />
    </DashboardLayout>
  );
}
