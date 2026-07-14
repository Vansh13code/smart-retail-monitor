import DashboardLayout from '../layouts/DashboardLayout';
import AnalysisPage from '../components/AnalysisPage';

export default function Product() {
  return (
    <DashboardLayout>
      <AnalysisPage title="Product Detection" endpoint="/detect-products/" />
    </DashboardLayout>
  );
}
