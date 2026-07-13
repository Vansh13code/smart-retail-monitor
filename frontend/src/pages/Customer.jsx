import DashboardLayout from "../layouts/DashboardLayout";
import AnalysisPage from "../components/AnalysisPage";

export default function Customer(){

return(

<DashboardLayout>

<AnalysisPage
  title="Customer Analysis"
  endpoint="/customer-analysis/"
/>

</DashboardLayout>

)

}
