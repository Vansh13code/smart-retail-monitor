import DashboardLayout from "../layouts/DashboardLayout";
import AnalysisPage from "../components/AnalysisPage";

export default function Inventory(){

return(

<DashboardLayout>

<AnalysisPage
  title="Inventory Analysis"
  endpoint="/inventory-analysis/"
/>

</DashboardLayout>

)

}
